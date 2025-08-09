# core/compressor.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import spacy
import sys
sys.path.append('..')
from utils import entropy

class MemoryCompressor(nn.Module):
    """
    基于信息熵与结构化知识的记忆压缩模块
    提取关键 token 并生成压缩后的嵌入表示
    """
    def __init__(self, language='en', hidden_size=384):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("nreimers/MiniLM-L6-H384-uncased")
        self.tokenizer = AutoTokenizer.from_pretrained("nreimers/MiniLM-L6-H384-uncased")
        self.nlp = spacy.load("en_core_web_sm" if language == 'en' else "zh_core_web_sm")
        self.entropy_calculator = entropy.WindowEntropyCalculator(hidden_size)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size + 1, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, text: str) -> dict:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        emb = self.encoder(**inputs).last_hidden_state  # [1, L, D]

        # 计算重要性
        entropy_score = self.entropy_calculator(emb)  # [1, L]
        importance = self.fusion(torch.cat([emb, entropy_score.unsqueeze(-1)], dim=-1))

        # 压缩：保留重要 token
        mask = importance > 0.5
        compressed = emb * mask.float().unsqueeze(-1)

        return {
            "compressed": compressed,
            "importance": importance,
            "tokens": inputs.input_ids
        }