import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GATConv
from torch_sparse import SparseTensor
from typing import List, Tuple
from transformers import DistilBertModel, DistilBertTokenizer, AutoModel, AutoTokenizer, AutoModelForCausalLM, \
    AutoModelForSequenceClassification, pipeline


class DifferentiableEntityMapper(nn.Module):
    """可导的实体映射组件"""

    def __init__(self, hidden_size):
        super().__init__()

        self.code_proj = nn.Linear(hidden_size, hidden_size)
        self.codebook = nn.Parameter(torch.randn(10000, hidden_size))

    def forward(self, emb) -> Tensor:
        """返回概率分布而不仅是索引"""

        code_emb = self.code_proj(emb)
        logits = torch.matmul(code_emb, self.codebook.T)  # [1, codebook_size]
        return torch.softmax(logits / 0.1, dim=-1)  # Gumbel softmax近似


class ParametricEdgeStore(nn.Module):
    """可学习边特征生成器"""

    def __init__(self, hidden_size):
        super().__init__()
        self.edge_net = nn.Sequential(
            nn.Linear(3 * hidden_size, 2 * hidden_size),
            nn.GELU(),
            nn.LayerNorm(2 * hidden_size),
            nn.Linear(2 * hidden_size, hidden_size)
        )

    def forward(self, src: Tensor, rel: Tensor, tgt: Tensor) -> Tensor:
        return self.edge_net(torch.cat([src, rel, tgt], dim=-1))


class DynamicGraphStorage(nn.Module):
    """动态图存储模块"""

    def __init__(self, hidden_size):
        super().__init__()
        self.node_emb = nn.EmbeddingBag(1000000, hidden_size, mode='mean')
        self.edge_store = ParametricEdgeStore(hidden_size)
        self.register_buffer('edge_index', torch.empty(2, 0, dtype=torch.long))
        self.gnn = GATConv(hidden_size, hidden_size, edge_dim=hidden_size)

    def add_edges(self, src_nodes: Tensor, tgt_nodes: Tensor, rel_embs: Tensor):
        """添加带关系的边"""
        edge_attrs = self.edge_store(src_nodes, rel_embs, tgt_nodes)
        new_edge_index = torch.stack([src_nodes, tgt_nodes], dim=0).to(rel_embs.device)
        self.edge_index = torch.cat([self.edge_index, new_edge_index], dim=1)
        return edge_attrs

    def forward(self, edge_attr, entity_count):
        node_feats = self.gnn(
            self.node_emb[:entity_count],
            self.edge_index[:, :edge_attr.size(0)],
            edge_attr=edge_attr
        )
        return node_feats


class TrainableGraphMemory(nn.Module):
    """完全可导的图记忆系统"""

    def __init__(self, hidden_size=384, max_entities=5000):
        super().__init__(hidden_size, max_entities)

        # 可导组件
        self.entity_mapper = DifferentiableEntityMapper(hidden_size)
        self.dynamic_store = DynamicGraphStorage(hidden_size)
        self.entity_encoder = nn.Linear(hidden_size, hidden_size)
        self.edge_attr = nn.Parameter(torch.empty(0, hidden_size), requires_grad=True)
        # 记忆重组增强网络
        self.relation_consolidator = nn.TransformerEncoderLayer(d_model=384, nhead=4)
        # 关系学习网络
        self.relation_finder = nn.Sequential(
            nn.Linear(hidden_size * 3, 256),
            nn.GELU(),
            nn.Linear(256, hidden_size))
        self.relation_scorer = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.SiLU(),
            nn.Linear(128, 1))

        # 记忆优化
        self.memory_consolidator = nn.LSTM(hidden_size, hidden_size)
        self.register_buffer('update_step', torch.tensor(0))

        self.register_buffer('entity_count', torch.tensor(0))
        # 初始化适配
        nn.init.xavier_uniform_(self.codebook)
        self.gnn = GATConv(hidden_size, hidden_size, edge_dim=hidden_size)

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        """端到端的记忆检索"""
        # 1. 查找相关实体
        entity_scores = torch.matmul(query, self.node_emb[:self.entity_count].T)
        seed_nodes = torch.topk(entity_scores, k=5, dim=-1).indices  # [batch, 5]

        # 2. 可微PPR计算
        ppr_scores = self.diff_ppr(seed_nodes)  # [batch, num_nodes]

        # 3. GNN增强
        node_feats = self.dynamic_store(self.edge_attr, self.entity_count)

        # 4. 重要性加权聚合
        return torch.matmul(ppr_scores.unsqueeze(1), node_feats).squeeze(1)  # [batch, hidden_size]

    def update_memory(self, compressed_embs: List[Tensor]):
        """完全可导的记忆更新流程"""
        batch_size = len(compressed_embs)

        # ===== 1. 实体聚类与节点分配 =====
        node_probs = [self.entity_mapper(emb) for emb in compressed_embs]
        node_ids = [torch.argmax(p) for p in node_probs]

        # Straight-through梯度估计
        hard_ids = torch.stack(node_ids)
        soft_probs = torch.stack(node_probs)
        node_ids = soft_probs + (hard_ids - soft_probs).detach()
        self.entity_count = torch.tensor(len(node_ids), device=compressed_embs[0].device)
        # ===== 2. 自监督关系发现 =====
        # 生成候选实体对
        candidate_pairs = []
        for i in range(batch_size):
            for j in range(i + 1, min(i + 3, len(compressed_embs))):  # 滑动窗口生成
                candidate_pairs.append((
                    compressed_embs[i],
                    compressed_embs[j],
                    torch.mean(compressed_embs[i:j + 1], dim=0)))

        # 关系特征提取
        rel_features = []
        for src, tgt, ctx in candidate_pairs:
            # 拼接[源实体, 目标实体, 上下文特征]
            rel_feat = self.relation_finder(
                torch.cat([src, tgt, ctx]))
            rel_features.append(rel_feat)

        # 关系重要性评分
        src_tgt = torch.stack([p[0] for p in candidate_pairs] + [p[1] for p in candidate_pairs])
        scores = self.relation_scorer(
            torch.cat([src_tgt[:len(candidate_pairs)], src_tgt[len(candidate_pairs):]], dim=-1))
        scores = torch.sigmoid(scores)

        # ===== 3. 动态关系筛选 =====
        # 基于熵的动态阈值
        entropy = -(scores * torch.log(scores + 1e-7)).mean()
        threshold = 0.5 + 0.2 * torch.tanh(entropy - 1.0)  # 自适应调节

        # 筛选重要关系
        mask = scores.squeeze() > threshold
        valid_relations = [candidate_pairs[i] for i in range(len(candidate_pairs)) if mask[i]]

        # ===== 4. 记忆系统更新 =====
        # 更新节点嵌入（带平滑更新机制）
        for idx, emb in zip(node_ids, compressed_embs):
            decay = 0.9 ** (self.update_step // 100)  # 指数衰减
            self.dynamic_store.node_emb.weight.data[idx] = \
                decay * self.dynamic_store.node_emb.weight.data[idx] + \
                (1 - decay) * emb

        # 添加新关系边
        new_edge_attrs = []
        for src_emb, tgt_emb, rel_emb in valid_relations:
            src_idx = self.entity_mapper(src_emb)
            tgt_idx = self.entity_mapper(tgt_emb)
            edge_attr = self.dynamic_store.edge_store(
                self.dynamic_store.node_emb(src_idx),
                rel_emb,
                self.dynamic_store.node_emb(tgt_idx))
            new_edge_attrs.append(edge_attr)

            # 双向添加关系
            self.dynamic_store.edge_index = torch.cat([
                self.dynamic_store.edge_index,
                torch.tensor([[src_idx, tgt_idx], [tgt_idx, src_idx]]).T
            ], dim=1)

        # 更新边属性
        if new_edge_attrs:
            self.edge_attr = nn.Parameter(
                torch.cat([self.edge_attr, torch.stack(new_edge_attrs)]),
                requires_grad=True)

        # ===== 5. 记忆巩固优化 =====
        if self.update_step % 100 == 0:
            self.consolidate_memory()

        self.update_step += 1

    def diff_ppr(self, seed_nodes: Tensor) -> Tensor:
        """稀疏矩阵优化的可导PPR"""
        num_nodes = self.entity_count.item()
        batch_size = seed_nodes.size(0)

        # 构造稀疏矩阵
        adj = SparseTensor(
            row=self.edge_index[0],
            col=self.edge_index[1],
            value=self.edge_attr,
            sparse_sizes=(num_nodes, num_nodes)
        )

        # 归一化处理
        row_sum = adj.sum(dim=1)
        adj = adj / row_sum.view(-1, 1)

        # 迭代计算（保持稀疏性）
        p = torch.zeros(batch_size, num_nodes,
                        device=seed_nodes.device).scatter(1, seed_nodes, 1.0)
        for _ in range(3):
            p = self.ppr_alpha * p + (1 - self.ppr_alpha) * adj @ p

        return p

    def consolidate_memory(self):
        """改进的记忆巩固方法"""
        # 1. 基于重要性的节点修剪
        importance = torch.norm(self.node_emb.weight, dim=1)
        mask = importance > 0.1  # 可调阈值

        # 2. 压缩存储
        self._prune_nodes(mask)
        self.edge_attr = self.relation_consolidator(self.edge_attr.unsqueeze(1)).squeeze(1)
        # 3. 边权重新校准
        self.edge_attr.data = torch.sigmoid(self.edge_attr)
