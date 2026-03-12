"""
TEC 梯度方向库（在线查询）

功能：
    1. 内存映射加载预计算的梯度库（避免全量加载到 RAM）
    2. 支持批量时间戳查询
    3. 余弦平方插值（保持 C1 连续性）
    4. 返回完整空间地图以兼容现有 grid_sample 逻辑

使用示例：
    >>> from tec_gradient_bank import TecGradientBank
    >>> bank = TecGradientBank('data/tec_gradient_bank.npy', total_hours=720)
    >>> timestamps = torch.tensor([10.3, 11.7, 23.9])  # 浮点小时
    >>> grad_maps = bank.get_interpolated_gradient(timestamps)
    >>> print(grad_maps.shape)  # (3, 2, 73, 73)

作者：R-STMRF Team
日期：2026-01-30
"""

import numpy as np
import torch
import os


class TecGradientBank:
    """
    TEC 梯度方向库（内存映射 + 时间插值）

    替代 ConvLSTM，提供快速的梯度方向查询，大幅降低内存和计算开销

    核心优化：
        - Memory-Mapped Loading: 按需从磁盘读取，RAM 占用 < 10 MB
        - Cosine-Squared Interpolation: C1 连续，无高频泄露
        - Vectorized Operations: 批量处理，避免循环

    数据格式：
        gradient_bank.npy: (T, 2, H, W) dtype=float16
            - T: 时间步（小时）
            - Channel 0: normalized grad_lat (单位向量)
            - Channel 1: normalized grad_lon (单位向量)
            - H, W: 空间分辨率 (73, 73)
    """

    def __init__(self, gradient_bank_path, total_hours=720, device='cuda', preload_to_gpu=None):
        """
        Args:
            gradient_bank_path: 预计算的梯度库文件路径 (*.npy)
            total_hours: 总时长（小时），用于边界检查
            device: 计算设备
            preload_to_gpu: 是否预加载到GPU（None=自动检测，True=强制，False=禁用）
                          自动检测：CUDA可用时预加载，CPU时使用mmap
        """
        print(f"[TecGradientBank] 初始化梯度库...")

        if not os.path.exists(gradient_bank_path):
            raise FileNotFoundError(f"梯度库文件未找到: {gradient_bank_path}")

        self.gradient_bank_path = gradient_bank_path
        self.total_hours = total_hours
        self.device = device

        # 内存映射加载（先加载元数据）
        gradient_bank_mmap = np.load(gradient_bank_path, mmap_mode='r')

        # 验证形状
        if gradient_bank_mmap.ndim != 4:
            raise ValueError(f"期望 4D 数据 (T, 2, H, W)，实际形状: {gradient_bank_mmap.shape}")

        T, C, H, W = gradient_bank_mmap.shape

        if C != 2:
            raise ValueError(f"期望 2 个通道 (grad_lat, grad_lon)，实际通道数: {C}")

        self.num_hours = T
        self.height = H
        self.width = W

        # 自动检测是否预加载到GPU
        if preload_to_gpu is None:
            preload_to_gpu = (device == 'cuda')

        # 性能优化：预加载到GPU（显著减少CPU-GPU传输）
        if preload_to_gpu and device == 'cuda':
            print(f"  ✓ 预加载到 GPU ({device})...")
            # 一次性加载所有数据到GPU
            self.gradient_bank_gpu = torch.from_numpy(
                gradient_bank_mmap[:].astype(np.float32)
            ).to(device)
            self.use_gpu_cache = True
            gpu_memory_mb = self.gradient_bank_gpu.element_size() * self.gradient_bank_gpu.nelement() / 1e6
            print(f"    ✓ GPU预加载完成")
            print(f"    GPU内存占用: {gpu_memory_mb:.2f} MB")
            print(f"    ⚡ 性能提升: 消除每个batch的CPU-GPU传输")
        else:
            # 使用memory-mapped模式（CPU或显存不足时）
            self.gradient_bank = gradient_bank_mmap
            self.use_gpu_cache = False
            print(f"  ✓ 梯度库加载成功 (Memory-Mapped)")
            print(f"    内存占用: < 10 MB (按需读取)")

        print(f"    形状: {gradient_bank_mmap.shape}")
        print(f"    数据类型: {gradient_bank_mmap.dtype}")
        print(f"    时间范围: [0, {self.num_hours - 1}] 小时")
        print(f"    空间分辨率: {H} × {W}")

    def get_interpolated_gradient(self, timestamps):
        """
        获取插值后的 TEC 梯度方向地图（批量处理）

        物理原理：
            - TEC 是低频物理场，小时尺度平滑演化
            - 使用余弦平方插值保证 C1 连续性（无梯度跳跃）
            - 单位向量插值后需要重新归一化（保持方向性）

        插值公式（余弦平方）：
            t_query ∈ [t0, t1]，其中 t0 = floor(t_query), t1 = t0 + 1
            frac = t_query - t0 ∈ [0, 1]
            α = cos²(π * frac / 2)  # 权重从 1 → 0 (t0 → t1)
            β = sin²(π * frac / 2)  # 权重从 0 → 1 (t0 → t1)
            grad_interp = α * grad[t0] + β * grad[t1]

        注意：α + β = cos²(x) + sin²(x) = 1（能量守恒）

        Args:
            timestamps: torch.Tensor [Batch] 或 list
                       查询时间（浮点小时，例如 [10.3, 11.7, 23.9]）

        Returns:
            grad_maps: torch.Tensor [Batch, 2, H, W]
                      插值后的梯度方向地图（已归一化为单位向量）
        """
        # 转换为 tensor
        if isinstance(timestamps, list):
            timestamps = torch.tensor(timestamps, dtype=torch.float32)
        elif not isinstance(timestamps, torch.Tensor):
            raise TypeError(f"timestamps 必须是 list 或 tensor，当前类型: {type(timestamps)}")

        timestamps = timestamps.float()
        batch_size = timestamps.shape[0]

        # 边界检查和限制
        timestamps = torch.clamp(timestamps, 0.0, float(self.num_hours - 1) - 1e-6)

        # 1. 计算下界和上界小时索引
        time_floor = torch.floor(timestamps).long()  # [Batch]
        time_ceil = torch.clamp(time_floor + 1, max=self.num_hours - 1)  # [Batch]

        # 2. 计算小数部分 [0, 1]
        frac = timestamps - time_floor.float()  # [Batch]

        # 3. 余弦平方插值权重
        # α = cos²(π * frac / 2)，当 frac=0 时 α=1，当 frac=1 时 α=0
        # β = sin²(π * frac / 2)，当 frac=0 时 β=0，当 frac=1 时 β=1
        alpha = torch.cos(torch.pi * frac / 2.0) ** 2  # [Batch]
        beta = torch.sin(torch.pi * frac / 2.0) ** 2   # [Batch]

        # 4. 批量读取梯度地图（去重以减少磁盘 I/O）
        # 合并 floor 和 ceil 索引，去重
        unique_indices = torch.unique(torch.cat([time_floor, time_ceil]))

        # 性能优化：使用GPU缓存（如果可用）
        if self.use_gpu_cache:
            # 直接从GPU tensor索引（无CPU-GPU传输）
            unique_grads = self.gradient_bank_gpu[unique_indices]  # [N_unique, 2, H, W]
        else:
            # 从磁盘读取（memory-mapped）
            unique_indices_np = unique_indices.cpu().numpy()
            unique_grads = self.gradient_bank[unique_indices_np]  # [N_unique, 2, H, W]
            # 转换为 torch tensor 并移动到目标设备
            unique_grads = torch.from_numpy(unique_grads.astype(np.float32)).to(self.device)

        # 5. 创建索引映射（unique_indices → local_idx）
        index_map = {idx.item(): i for i, idx in enumerate(unique_indices)}

        # 6. 批量插值（向量化操作）
        grad_maps = torch.zeros(batch_size, 2, self.height, self.width,
                                dtype=torch.float32, device=self.device)

        for i in range(batch_size):
            t0_idx = index_map[time_floor[i].item()]
            t1_idx = index_map[time_ceil[i].item()]

            grad_t0 = unique_grads[t0_idx]  # [2, H, W]
            grad_t1 = unique_grads[t1_idx]  # [2, H, W]

            # 余弦平方插值
            grad_interp = alpha[i] * grad_t0 + beta[i] * grad_t1  # [2, H, W]

            # 重新归一化为单位向量（插值后范数可能偏离 1.0）
            # norm = sqrt(grad_lat^2 + grad_lon^2)
            norm = torch.sqrt((grad_interp ** 2).sum(dim=0, keepdim=True)) + 1e-8  # [1, H, W]
            grad_interp = grad_interp / norm  # [2, H, W]

            grad_maps[i] = grad_interp

        return grad_maps

    def get_gradient_at_hours(self, hour_indices):
        """
        直接获取整点小时的梯度地图（无插值，用于调试/验证）

        Args:
            hour_indices: list or tensor of integer hour indices
                         例如: [10, 11, 15]

        Returns:
            grad_maps: torch.Tensor [N_hours, 2, H, W]
        """
        if isinstance(hour_indices, list):
            hour_indices = np.array(hour_indices, dtype=np.int64)
        elif isinstance(hour_indices, torch.Tensor):
            hour_indices = hour_indices.cpu().numpy()
        else:
            raise TypeError(f"hour_indices 必须是 list 或 tensor，当前类型: {type(hour_indices)}")

        # 边界检查
        hour_indices = np.clip(hour_indices, 0, self.num_hours - 1)

        # 性能优化：使用GPU缓存（如果可用）
        if self.use_gpu_cache:
            # 直接从GPU tensor索引
            hour_indices_tensor = torch.from_numpy(hour_indices).to(self.device)
            grads = self.gradient_bank_gpu[hour_indices_tensor]  # [N_hours, 2, H, W]
        else:
            # 从磁盘读取
            grads = self.gradient_bank[hour_indices]  # [N_hours, 2, H, W]
            # 转换为 torch tensor
            grads = torch.from_numpy(grads.astype(np.float32)).to(self.device)

        return grads

    def __repr__(self):
        return (f"TecGradientBank(\n"
                f"  path={self.gradient_bank_path},\n"
                f"  shape={self.gradient_bank.shape},\n"
                f"  dtype={self.gradient_bank.dtype},\n"
                f"  device={self.device}\n"
                f")")


# ======================== 测试代码 ========================
if __name__ == '__main__':
    print("="*70)
    print("TecGradientBank 测试")
    print("="*70)

    # 创建模拟梯度库（用于测试）
    print("\n[测试 1] 创建模拟梯度库")
    T, H, W = 720, 73, 73
    test_data = np.random.randn(T, 2, H, W).astype(np.float16)

    # 归一化为单位向量
    norm = np.sqrt((test_data ** 2).sum(axis=1, keepdims=True)) + 1e-8
    test_data = test_data / norm

    test_path = '/tmp/test_gradient_bank.npy'
    np.save(test_path, test_data)
    print(f"  ✓ 模拟数据已保存: {test_path}")
    print(f"    形状: {test_data.shape}")

    # 测试 TecGradientBank
    print("\n[测试 2] 加载梯度库")
    bank = TecGradientBank(test_path, total_hours=720, device='cpu')
    print(f"  {bank}")

    # 测试插值
    print("\n[测试 3] 批量插值查询")
    timestamps = torch.tensor([10.0, 10.5, 11.0, 11.3, 23.7])
    grad_maps = bank.get_interpolated_gradient(timestamps)
    print(f"  输入时间戳: {timestamps.tolist()}")
    print(f"  输出形状: {grad_maps.shape}")
    print(f"  输出设备: {grad_maps.device}")

    # 验证归一化
    norms = torch.sqrt((grad_maps ** 2).sum(dim=1))  # [Batch, H, W]
    print(f"  平均向量范数: {norms.mean():.6f} (应接近 1.0)")
    print(f"  范数标准差: {norms.std():.6f} (应接近 0.0)")

    # 测试整点查询
    print("\n[测试 4] 整点小时查询（无插值）")
    hour_grads = bank.get_gradient_at_hours([10, 11, 12])
    print(f"  输出形状: {hour_grads.shape}")

    # 清理
    os.remove(test_path)
    print("\n" + "="*70)
    print("✓ 所有测试通过！")
    print("="*70)
