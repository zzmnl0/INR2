"""
R-STMRF (Recurrent Spatio-Temporal Modulated Residual Field) 主模型

物理引导的循环时空调制残差场

核心架构：
    Ne = IRI_frozen + Decoder(h_spatial_modulated, h_temporal_modulated)

其中：
    - h_spatial: SIREN 空间基函数，受 TEC 地图调制（FiLM）
    - h_temporal: SIREN 时间基函数，受 Kp/F10.7 调制（Additive Shift）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from .siren_layers import ModulatedSIRENNet
from .recurrent_parts import GlobalEnvEncoder


class R_STMRF_Model(nn.Module):
    """
    R-STMRF 主模型

    分支 A: 空间分支（受 TEC 调制）
    分支 B: 时间分支（受 Kp/F10.7 调制）
    分支 C: 物理约束（Loss）
    """

    def __init__(self, iri_proxy, lat_range, lon_range, alt_range,
                 sw_manager=None, tec_manager=None, start_date_str=None, config=None):
        """
        Args:
            iri_proxy: IRI 神经代理场（冻结）
            lat_range: 纬度范围
            lon_range: 经度范围
            alt_range: 高度范围
            sw_manager: 空间天气管理器
            tec_manager: TEC 管理器（保留用于兼容性，实际不使用）
            start_date_str: 起始日期字符串
            config: 配置字典
        """
        super().__init__()

        # 物理参数
        self.lat_range = lat_range
        self.lon_range = lon_range
        self.alt_range = alt_range
        self.total_hours = config.get('total_hours', 720.0)
        self.seq_len = config['seq_len']

        # 数据管理器
        self.sw_manager = sw_manager
        self.tec_manager = tec_manager
        self.start_date = pd.to_datetime(start_date_str) if start_date_str else None

        # IRI 神经代理场（冻结）
        self.iri_proxy = iri_proxy
        for param in self.iri_proxy.parameters():
            param.requires_grad = False
        self.iri_proxy.eval()

        # 模型维度
        self.basis_dim = config.get('basis_dim', 64)
        self.siren_hidden = config.get('siren_hidden', 128)
        self.siren_layers = config.get('siren_layers', 3)
        self.omega_0 = config.get('omega_0', 30.0)

        # 环境特征维度
        self.env_hidden_dim = config.get('env_hidden_dim', 64)

        # ==================== 分支 A: 空间分支（主路）====================
        # 空间基函数网络 (SIREN) - 主建模网络
        # 输入: (Lat, Lon, Alt, sin_lt, cos_lt) = 5维
        # 职责: 学习电子密度的空间表达能力（无时序约束）
        self.spatial_basis_net = ModulatedSIRENNet(
            in_features=5,
            hidden_features=self.siren_hidden,
            hidden_layers=self.siren_layers,
            out_features=self.basis_dim,
            omega_0=self.omega_0
        )

        # ==================== TEC 梯度方向（外部输入）====================
        # 新架构: TEC 梯度方向通过离线预计算 + 时间插值获得（TecGradientBank）
        # - 移除 ConvLSTM (SpatialContextEncoder)
        # - 移除 tec_gradient_direction_head
        # - TEC 梯度方向作为 forward() 的外部输入传入
        # 优势: 消除在线 ConvLSTM 计算，大幅降低内存和训练时间

        # ==================== 分支 B: 时间分支 ====================
        # 时间基函数网络 (SIREN)
        # 输入: Time = 1维
        self.temporal_basis_net = ModulatedSIRENNet(
            in_features=1,
            hidden_features=self.siren_hidden,
            hidden_layers=self.siren_layers,
            out_features=self.basis_dim,
            omega_0=self.omega_0
        )

        # 全局环境编码器 (LSTM)
        self.env_encoder = GlobalEnvEncoder(
            input_dim=2,
            hidden_dim=self.env_hidden_dim,
            num_layers=2,
            dropout=0.1
        )

        # 时间调制头（Additive Shift: h + beta）
        self.temporal_modulation_head = nn.Sequential(
            nn.Linear(self.env_hidden_dim, self.basis_dim),
            nn.Tanh()
        )

        # ==================== 融合解码器 ====================
        # 输入: concat(h_spatial_mod, h_temporal_mod) = 2 * basis_dim
        self.decoder = nn.Sequential(
            nn.Linear(self.basis_dim * 2, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

        # 不确定性估计头
        self.uncertainty_head = nn.Sequential(
            nn.Linear(self.basis_dim * 2, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

        # 残差缩放参数
        self.resid_scale = nn.Parameter(torch.tensor(1.0))

        # 初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """权重初始化"""
        # 解码器最后一层零初始化（残差学习）
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

        # 不确定性头零初始化
        nn.init.zeros_(self.uncertainty_head[-1].weight)
        nn.init.zeros_(self.uncertainty_head[-1].bias)

    def normalize_coords_internal(self, lat, lon, alt, time):
        """INR 内部坐标归一化"""
        lat_n = lat / 90.0
        lon_n = lon / 180.0

        alt_min, alt_max = self.alt_range
        alt_n = 2.0 * (alt - alt_min) / (alt_max - alt_min) - 1.0

        time_n = (time / self.total_hours) * 2.0 - 1.0

        return lat_n, lon_n, alt_n, time_n

    def get_background(self, lat, lon, alt, time):
        """
        查询 IRI Neural Proxy 背景值

        关键：
        1. IRI Proxy 参数 frozen，但需要对输入 coords 保留梯度
        2. 临时切换到 train 模式以启用梯度传播
        """
        coords = torch.stack([lat, lon, alt, time], dim=-1)

        # 临时启用 train 模式让梯度通过
        was_training = self.iri_proxy.training
        self.iri_proxy.train()

        try:
            background_log = self.iri_proxy(coords)
        finally:
            # 恢复原始状态
            if not was_training:
                self.iri_proxy.eval()

        return background_log

    def create_temporal_mask(self, current_time_batch):
        """生成时序掩码 (t < 0 为无效)"""
        device = current_time_batch.device
        offsets = torch.arange(self.seq_len - 1, -1, -1, device=device)
        history_times = current_time_batch.unsqueeze(1) - offsets.unsqueeze(0)
        return history_times < 0


    def forward(self, coords, sw_seq, tec_grad_direction):
        """
        前向传播（新架构：接受预计算的 TEC 梯度方向）

        Args:
            coords: [Batch, 4] (Lat, Lon, Alt, Time)
            sw_seq: [Batch, Seq, 2] 空间天气序列 (Kp, F10.7)
            tec_grad_direction: [Batch, 2, H, W] 预计算的 TEC 梯度方向（单位向量）
                               - Channel 0: normalized grad_lat
                               - Channel 1: normalized grad_lon
                               - 通过 TecGradientBank 离线预计算 + 时间插值获得

        Returns:
            output: [Batch, 1] 预测 Ne (对数尺度)
            log_var: [Batch, 1] 不确定性 (对数方差)
            final_correction: [Batch, 1] 最终残差
            extra_outputs: dict 附加输出（用于分析）
        """
        lat, lon, alt, time = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
        batch_size = lat.shape[0]

        # 1. 获取 IRI 背景
        background = self.get_background(lat, lon, alt, time)

        # 2. 坐标归一化
        lat_n, lon_n, alt_n, time_n = self.normalize_coords_internal(lat, lon, alt, time)

        # 3. 计算地方时特征
        local_time_hour = (time % 24.0) + (lon / 15.0)
        lt_norm = local_time_hour / 24.0
        sin_lt = torch.sin(2 * np.pi * lt_norm)
        cos_lt = torch.cos(2 * np.pi * lt_norm)

        # 4. 时序掩码
        temporal_mask = self.create_temporal_mask(time)

        # ==================== 分支 A: 空间分支 ====================
        # A1. 生成空间基函数（主建模网络）
        spatial_input = torch.stack([lat_n, lon_n, alt_n, sin_lt, cos_lt], dim=1)
        h_spatial = self.spatial_basis_net(spatial_input)  # [Batch, basis_dim]
        # h_spatial 表示"在无时序约束下的电子密度空间基函数"

        # A2. TEC 梯度方向（外部输入，无需在线计算）
        # tec_grad_direction: [Batch, 2, H, W] - 预计算的梯度方向场
        # 通过 TecGradientBank 离线预计算 + 余弦平方插值获得
        # 仅用于物理损失约束，不参与前向传播的数值计算

        # A3. 空间分支不使用 TEC 调制，直接使用原始 SIREN 输出
        # TEC 仅通过物理损失函数约束梯度方向一致性
        h_spatial_mod = h_spatial  # [Batch, basis_dim] - 无调制

        # ==================== 分支 B: 时间分支 ====================
        # B1. 生成时间基函数
        temporal_input = time_n.unsqueeze(1)  # [Batch, 1]
        h_temporal = self.temporal_basis_net(temporal_input)  # [Batch, basis_dim]

        # B2. 环境上下文编码（LSTM）
        z_env = self.env_encoder(sw_seq, mask=temporal_mask)  # [Batch, env_hidden_dim]

        # B3. 时间调制（Additive Shift: h + beta）
        beta_temporal = self.temporal_modulation_head(z_env)  # [Batch, basis_dim]
        h_temporal_mod = h_temporal + beta_temporal  # [Batch, basis_dim]

        # ==================== 融合解码 ====================
        # 拼接调制后的特征
        fusion_features = torch.cat([h_spatial_mod, h_temporal_mod], dim=-1)  # [Batch, 2*basis_dim]

        # 解码得到残差
        raw_correction = self.decoder(fusion_features)  # [Batch, 1]
        raw_correction = torch.tanh(self.resid_scale * raw_correction)

        # 最终输出
        output = background + raw_correction

        # 不确定性估计
        raw_log_var = self.uncertainty_head(fusion_features)
        log_var = torch.clamp(raw_log_var, -10.0, 10.0)

        # 附加输出（用于分析和可视化）
        extra_outputs = {
            'background': background,
            'h_spatial': h_spatial,
            'h_spatial_mod': h_spatial_mod,
            'h_temporal': h_temporal,
            'h_temporal_mod': h_temporal_mod,
            'beta_temporal': beta_temporal,
            'z_env': z_env,
            'tec_grad_direction': tec_grad_direction,  # [Batch, 2, H, W] - TEC 梯度方向
            'coords_normalized': torch.stack([lat_n, lon_n], dim=1),  # [Batch, 2] - 用于采样
        }

        return output, log_var, raw_correction, extra_outputs


# ======================== 测试代码 ========================
if __name__ == '__main__':
    print("="*60)
    print("R-STMRF 模型测试")
    print("="*60)

    # 模拟配置
    config = {
        'total_hours': 720.0,
        'seq_len': 6,
        'basis_dim': 64,
        'siren_hidden': 128,
        'siren_layers': 3,
        'omega_0': 30.0,
        'env_hidden_dim': 64,
    }

    # 模拟 IRI proxy（简单线性层）
    class DummyIRIProxy(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(4, 1)

        def forward(self, coords):
            return self.net(coords)

    iri_proxy = DummyIRIProxy()

    # 创建模型
    model = R_STMRF_Model(
        iri_proxy=iri_proxy,
        lat_range=(-90, 90),
        lon_range=(-180, 180),
        alt_range=(120, 500),
        sw_manager=None,
        tec_manager=None,
        start_date_str='2024-09-01 00:00:00',
        config=config
    )

    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 测试前向传播
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    batch_size = 16
    coords = torch.randn(batch_size, 4).to(device)  # [Batch, 4]
    coords[:, 0] = coords[:, 0] * 90  # Lat
    coords[:, 1] = coords[:, 1] * 180  # Lon
    coords[:, 2] = 200 + coords[:, 2] * 100  # Alt
    coords[:, 3] = coords[:, 3].abs() * 100  # Time

    sw_seq = torch.randn(batch_size, 6, 2).to(device)  # [Batch, Seq=6, 2]

    # 模拟预计算的 TEC 梯度方向（新架构）
    tec_grad_direction = torch.randn(batch_size, 2, 73, 73).to(device)  # [Batch, 2, H, W]
    # 归一化为单位向量
    norm = torch.sqrt((tec_grad_direction ** 2).sum(dim=1, keepdim=True)) + 1e-8
    tec_grad_direction = tec_grad_direction / norm

    print("\n输入形状:")
    print(f"  coords: {coords.shape}")
    print(f"  sw_seq: {sw_seq.shape}")
    print(f"  tec_grad_direction: {tec_grad_direction.shape}")

    # 前向传播
    output, log_var, correction, extras = model(coords, sw_seq, tec_grad_direction)

    print("\n输出形状:")
    print(f"  output: {output.shape}")
    print(f"  log_var: {log_var.shape}")
    print(f"  correction: {correction.shape}")

    print("\n附加输出:")
    for key, val in extras.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: {val.shape}")

    # 测试梯度流动
    loss = output.sum()
    loss.backward()
    print("\n梯度测试:")
    print(f"  Spatial Basis 梯度: {model.spatial_basis_net.siren.net[0].linear.weight.grad.norm().item():.6f}")
    print(f"  Temporal Basis 梯度: {model.temporal_basis_net.siren.net[0].linear.weight.grad.norm().item():.6f}")

    print("\n" + "="*60)
    print("所有测试通过!")
    print("="*60)
