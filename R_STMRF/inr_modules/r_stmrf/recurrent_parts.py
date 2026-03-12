"""
循环神经网络组件

包含：
    GlobalEnvEncoder: LSTM 编码器，处理 Kp/F10.7 全局环境序列

注意：
    - ConvLSTM 和 SpatialContextEncoder 已在 v2.0 架构中移除
    - TEC 梯度方向现在通过 TecGradientBank 离线预计算获得
"""

import torch
import torch.nn as nn


# ======================== 全局环境编码器 (LSTM) ========================
class GlobalEnvEncoder(nn.Module):
    """
    全局环境编码器（处理 Kp/F10.7）

    输入: [Batch, Seq_Len, 2] (Kp, F10.7)
    输出: [Batch, Hidden_Dim] 全局环境上下文向量

    作用:
        - 提取磁暴和太阳活动的时序演变特征
        - 用于时间基函数的加性调制（Shift）
    """

    def __init__(self, input_dim=2, hidden_dim=64, num_layers=2, dropout=0.1):
        """
        Args:
            input_dim: 输入维度（Kp + F10.7 = 2）
            hidden_dim: LSTM 隐层维度
            num_layers: LSTM 层数
            dropout: Dropout 概率
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM 编码器
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )

        # 输出投影（取最后时刻的隐状态）
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

    def forward(self, sw_seq, mask=None):
        """
        前向传播

        Args:
            sw_seq: [Batch, Seq_Len, 2] Kp/F10.7 序列
            mask: [Batch, Seq_Len] 可选的时序掩码（True 表示无效）

        Returns:
            z_env: [Batch, Hidden_Dim] 全局环境上下文向量
        """
        # 如果有掩码，将无效位置置零
        if mask is not None:
            sw_seq = sw_seq.masked_fill(mask.unsqueeze(-1), 0.0)

        # LSTM 编码
        # output: [Batch, Seq, Hidden_Dim]
        # h_n: [Num_Layers, Batch, Hidden_Dim]
        output, (h_n, c_n) = self.lstm(sw_seq)

        # 取最后一层的最后时刻隐状态
        z_env = h_n[-1]  # [Batch, Hidden_Dim]

        # 投影
        z_env = self.output_proj(z_env)

        return z_env


# ======================== 测试代码 ========================
if __name__ == '__main__':
    print("="*60)
    print("GlobalEnvEncoder 测试")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 测试 GlobalEnvEncoder
    print("\n[测试] GlobalEnvEncoder (LSTM)")
    env_encoder = GlobalEnvEncoder(input_dim=2, hidden_dim=64, num_layers=2).to(device)
    sw_seq = torch.randn(16, 6, 2).to(device)  # [Batch=16, Seq=6, Feat=2]
    z_env = env_encoder(sw_seq)
    print(f"  输入: {sw_seq.shape}")
    print(f"  输出: {z_env.shape}")
    print(f"  参数量: {sum(p.numel() for p in env_encoder.parameters()):,}")

    # 测试梯度流动
    print("\n[测试] 梯度流动")
    loss = z_env.sum()
    loss.backward()
    print(f"  LSTM 梯度范数: {env_encoder.lstm.weight_ih_l0.grad.norm().item():.6f}")

    print("\n" + "="*60)
    print("测试通过!")
    print("="*60)
