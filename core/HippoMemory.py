# HippoMemory.py
from compressor import *
from hippocampus import *
from graph_memory import *
from router import *
import torch

# --- 在 code.txt 文件的相应位置替换或添加以下代码 ---

import torch
import torch.nn as nn

class HippoMemorySystem(nn.Module):
    """
    类脑记忆增强系统
    为大模型提供长期上下文记忆能力
    继承自 nn.Module，可被训练和保存
    """

    def __init__(self, language='en', hidden_size=384):
        """
        初始化 HippoMemorySystem
        :param language: 用于 NLP 处理的语言 ('en' or 'zh')
        :param hidden_size: 模型隐藏层维度，需与子模块匹配
        """
        super(HippoMemorySystem, self).__init__()
        self.hidden_size = hidden_size
        self.language = language

        # 1. 注册子模块 (nn.Modules)
        # 确保子模块的 __init__ 方法接受 hidden_size 参数
        self.compressor = MemoryCompressor(language=language, hidden_size=hidden_size)
        # 注意：HippocampalGate 的 __init__ 需要 hidden_size
        self.hippocampus = HippocampalGate(hidden_size=hidden_size, capacity=512)
        self.graph_memory = GraphMemory(hidden_size=hidden_size)
        self.router = DynamicRouter(hidden_size=hidden_size)

        # 2. 注册缓冲区 (Buffers) 用于存储状态
        # 短期记忆池
        self.register_buffer('short_term_memory', torch.zeros(1, 0, self.hidden_size))
        # 上次更新时间戳
        self.register_buffer('last_update', torch.tensor(0.0))

    def memory_update(self, text: str, timestamp: float = None):
        """
        更新短期和长期记忆
        :param text: 输入的文本
        :param timestamp: 可选的时间戳（秒）。如果不提供，将使用内部计数器递增。
        """
        # 计算时间差
        if timestamp is None:
            # 如果没有提供外部时间戳，使用内部计数器模拟时间流逝
            time_delta = 1.0
            self.last_update += 1.0
        else:
            time_delta = timestamp - self.last_update.item()
            # 更新内部时间戳
            self.last_update.fill_(timestamp)  # 使用 fill_ 更新 buffer 的值

        memory_info = self.compressor(text)
        compressed_emb = memory_info['compressed']  # [1, seq_len, hidden_size]
        # importance = memory_info['importance'] # 可选使用

        updated_short_mem = self.hippocampus(compressed_emb, time_delta)

        self.short_term_memory = updated_short_mem.detach()  # detach 避免梯度回传到历史记忆

        doc = self.compressor.nlp(text) # 获取 spaCy 文档
        entities = [ent.text for ent in doc.ents]
        if len(entities) >= 2:
            head_text = entities[0]
            tail_text = entities[1]
            # 简单地用平均嵌入作为关系
            rel_emb = compressed_emb.mean(dim=1) # [1, hidden_size]

            head_id = hash(head_text) % 10000 # 简单哈希映射到 Embedding 范围
            tail_id = hash(tail_text) % 10000
            self.graph_memory.add_triplet(head_id, rel_emb, tail_id)
        # --- 示例结束 ---

    def forward(self, current_emb: torch.Tensor) -> torch.Tensor:
        """
        根据当前输入和内部存储的记忆生成增强的上下文表示
        :param current_emb: 当前输入的嵌入 [batch_size, seq_len, hidden_size]
        :return: 增强后的上下文表示 [batch_size, seq_len, hidden_size]
        """

        query_emb = current_emb.mean(dim=1)  # [batch_size, hidden_size]
        long_term_mem = self.graph_memory(query_emb)  # [batch_size, hidden_size]
        # 为了后续处理方便，扩展维度
        long_term_mem = long_term_mem.unsqueeze(1)  # [batch_size, 1, hidden_size]


        if self.short_term_memory.size(1) > 0:
            short_term_rep = self.short_term_memory.mean(dim=1)  # [1, hidden_size]
        else:
            short_term_rep = torch.zeros(current_emb.size(0), self.hidden_size,
                                         device=current_emb.device)  # [1, hidden_size]
        short_term_rep = short_term_rep.expand(current_emb.size(0), -1).unsqueeze(1)  # [batch_size, 1, hidden_size]

        current_rep = current_emb.mean(dim=1, keepdim=True)  # [batch_size, 1, hidden_size]
        short_rep = short_term_rep  # [batch_size, 1, hidden_size]
        long_rep = long_term_mem  # [batch_size, 1, hidden_size]

        fused_rep = self.router(current_rep.squeeze(1), short_rep.squeeze(1),
                                long_rep.squeeze(1))  # [batch_size, hidden_size]
        augmented_emb = fused_rep.unsqueeze(1).repeat(1, current_emb.size(1), 1)  # [B, S, H]

        return augmented_emb

    def reset(self):
        """重置记忆状态，开始新的对话"""
        self.short_term_memory = torch.zeros_like(self.short_term_memory)
        self.last_update.zero_()  # 重置时间戳




if __name__ =='__main__':
    hippo_memory = HippoMemorySystem()
    torch.save(hippo_memory.state_dict(), 'hippo_memory_system.pth')