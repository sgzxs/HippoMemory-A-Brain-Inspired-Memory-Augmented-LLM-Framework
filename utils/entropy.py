import torch.nn as nn
import torch
from typing import Tuple, Dict
import torch.nn.functional as F

class WindowEntropyCalculator(nn.Module):
    """集成多维度评估的熵计算器"""

    def __init__(self, hidden_size: int, max_window: int = 16):
        super().__init__()
        assert max_window % 2 == 0, "max_window应为偶数"

        # 基础熵计算配置
        self.max_window = int(max_window)
        self.stride = int(max_window // 2)
        self.hidden_size = hidden_size

        # 1. 改进的窗口计算（带语法感知的分组卷积）
        self.syntax_aware_conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=self.max_window,
            stride=self.stride,
            groups=8,  # 部分分组捕获语法模式
            bias=False
        )
        nn.init.normal_(self.syntax_aware_conv.weight, mean=0, std=0.02)
        self.syntax_aware_conv.weight.requires_grad_(False)
        # 2. 熵特征增强模块
        self.entropy_enhancer = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

        # 3. 动态温度调控
        self.temperature_predictor = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()
        )
        self.min_temperature = 0.01

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        输入: [batch_size, seq_len, hidden_size]
        输出:
            - importance: [batch_size, seq_len] (重要性分数)
            - threshold: [batch_size, 1] (动态阈值)
        """
        batch_size, seq_len, _ = x.shape

        # ===== 1. 语法增强的窗口计算 =====
        x = x.permute(0, 2, 1)  # [batch, hidden, seq]
        pad_len = (self.max_window - seq_len % self.stride) % self.stride
        x_padded = F.pad(x, (0, pad_len), mode='reflect')

        # 语法敏感的特征提取
        unfolded = self.syntax_aware_conv(x_padded)  # [batch, hidden, num_windows]

        # ===== 2. 动态温度调控 =====
        global_feat = x.mean(dim=-1)  # [batch, hidden]
        temperature = self.temperature_predictor(global_feat) + self.min_temperature

        # ===== 3. 熵计算与增强 =====
        probs = F.softmax(unfolded / temperature.unsqueeze(-1), dim=-1)
        window_entropy = -torch.sum(probs * torch.log(probs + 1e-7), dim=-1)

        # 熵特征增强
        enhanced_entropy = self.entropy_enhancer(
            window_entropy.unsqueeze(-1)  # [batch, hidden, num_windows, 1]
        ).squeeze(-1)

        # ===== 4. 重建序列重要性 =====
        entropy_score = F.fold(
            enhanced_entropy.unsqueeze(2),
            output_size=(1, seq_len + pad_len),
            kernel_size=(1, self.max_window),
            stride=(1, self.stride)
        )[:, 0, 0, :seq_len]  # [batch, seq_len]
        # 计算归一化权重矩阵
        ones = torch.ones_like(window_entropy)  # 全 1 张量
        weight_matrix = F.fold(
            F.unfold(
                ones.unsqueeze(1).unsqueeze(2),  # 调整形状为 [batch, 1, hidden, num_windows]
                kernel_size=(1, self.max_window),
                stride=(1, self.stride)
            ),
            output_size=(1, seq_len + pad_len),
            kernel_size=(1, self.max_window),
            stride=(1, self.stride)
        )[:, 0, 0, :seq_len]  # [batch, seq_len]

        normalized_entropy = entropy_score / weight_matrix.clamp(min=1e-6)
        normalized_entropy = (normalized_entropy - normalized_entropy.min(dim=1, keepdim=True)[0]) / \
                             (normalized_entropy.max(dim=1, keepdim=True)[0] -
                              normalized_entropy.min(dim=1, keepdim=True)[0] + 1e-7)

        return 1 - normalized_entropy
