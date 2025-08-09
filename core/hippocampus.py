# core/hippocampus.py
import torch
import torch.nn as nn

class HippocampalGate(nn.Module):
    """
    类海马体门控机制
    控制短期记忆的更新、遗忘与重组
    """
    def __init__(self, hidden_size=384, capacity=512):
        super().__init__()
        self.capacity = capacity
        self.hidden_size = hidden_size
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.GELU(),
            nn.Linear(256, 3),  # retain, compress, forget
            nn.Softmax(dim=-1)
        )
        self.time_decay = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, hidden_size),
            nn.Sigmoid()
        )
        self.register_buffer('short_term_memory', torch.zeros(1, 0, hidden_size))
        self.register_buffer('last_update', torch.tensor(0.0))

    def forward(self, new_memory: torch.Tensor, time_delta: float):
        """
        new_memory: [1, L, D] 来自压缩器的输出
        time_delta: 距上次更新的时间（秒）
        """
        B, L, D = new_memory.shape
        decay = self.time_decay(torch.tensor([[time_delta]]))  # [1, D]
        new_memory = new_memory * decay.unsqueeze(1)

        if self.short_term_memory.size(1) == 0:
            updated = new_memory
        else:
            gate = self.gate(torch.cat([
                self.short_term_memory.mean(1),
                new_memory.mean(1)
            ], dim=-1))  # [1, 3]

            retain_mask = gate[:, 0:1].unsqueeze(-1)  # 保留旧记忆
            forget_mask = gate[:, 2:3].unsqueeze(-1)  # 遗忘

            retained = self.short_term_memory * (1 - forget_mask) * retain_mask
            updated = torch.cat([retained, new_memory], dim=1)

        # 容量控制
        if updated.size(1) > self.capacity:
            updated = updated[:, -self.capacity:]

        self.short_term_memory = updated.detach()  # 不反向传播到历史记忆
        return updated