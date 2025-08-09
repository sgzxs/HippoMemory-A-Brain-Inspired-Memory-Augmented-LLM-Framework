# core/graph_memory.py
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

class GraphMemory(nn.Module):
    """
    基于图神经网络的长期记忆系统
    支持实体、关系抽取与记忆检索
    """
    def __init__(self, hidden_size=384):
        super().__init__()
        self.hidden_size = hidden_size
        self.gnn = GATConv(hidden_size, hidden_size, edge_dim=hidden_size)
        self.node_embeddings = nn.Embedding(10000, hidden_size)
        self.edge_encoder = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU()
        )
        self.register_buffer('edge_index', torch.empty(2, 0, dtype=torch.long))
        self.edge_attrs = []

    def add_triplet(self, head_id, rel_emb, tail_id):
        """添加三元组"""
        edge_idx = torch.tensor([[head_id, tail_id], [tail_id, head_id]])
        self.edge_index = torch.cat([self.edge_index, edge_idx], dim=1)
        edge_attr = self.edge_encoder(torch.cat([head_id, rel_emb, tail_id]))
        self.edge_attrs.append(edge_attr)

    def forward(self, query_emb):
        """检索相关记忆"""
        if len(self.edge_attrs) == 0:
            return torch.zeros_like(query_emb)
        x = self.node_embeddings.weight
        edge_attr = torch.stack(self.edge_attrs)
        out = self.gnn(x, self.edge_index, edge_attr=edge_attr)
        sim = torch.cosine_similarity(query_emb, out, dim=-1)
        return out[sim.topk(1).indices].mean(0)