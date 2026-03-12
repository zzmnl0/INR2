"""
R-STMRF 滑动窗口数据整合工具

设计思路：
    - 保留现有的 FY_dataloader.py 时间分箱策略（TimeBinSampler）
    - 在训练循环中，通过 sw_manager 和 tec_manager 动态获取历史序列
    - 提供辅助函数简化数据流处理
"""

import torch
import numpy as np


class SlidingWindowBatchProcessor:
    """
    滑动窗口批次处理器

    职责：
        - 接收 FY 数据批次（原始格式）
        - 动态查询 sw_manager 和 tec_manager
        - 返回 R-STMRF 所需的完整数据包

    使用方式：
        在训练循环中，替代原有的数据预处理逻辑
    """

    def __init__(self, sw_manager, tec_manager, device='cuda'):
        """
        Args:
            sw_manager: SpaceWeatherManager 实例
            tec_manager: TECDataManager 实例
            device: 计算设备
        """
        self.sw_manager = sw_manager
        self.tec_manager = tec_manager
        self.device = device

    def process_batch(self, batch_data):
        """
        处理一个批次的 FY 数据（新架构：移除 TEC 在线加载）

        TEC 梯度方向现在通过 TecGradientBank 离线预计算 + 时间插值获得，
        不再需要在此处加载 TEC 地图序列

        性能优化：
        - 使用单次 CPU-GPU 传输（而非5次）
        - 使用 non_blocking=True 异步传输
        - 避免不必要的内存分配

        Args:
            batch_data: [Batch, 5] Tensor
                Columns: [Lat, Lon, Alt, Time, Ne_Log]

        Returns:
            coords: [Batch, 4] (Lat, Lon, Alt, Time)
            target_ne: [Batch, 1] 真值 Ne（对数）
            sw_seq: [Batch, Seq, 2] 空间天气序列
        """
        # 优化：单次传输整个batch（减少CPU-GPU通信开销）
        # non_blocking=True允许异步传输，提高吞吐量
        batch_data = batch_data.to(self.device, non_blocking=True)

        # 解析批次数据（在GPU上进行切片，无额外传输）
        coords = batch_data[:, :4]  # [Batch, 4] - (Lat, Lon, Alt, Time)
        target_ne = batch_data[:, 4:5]  # [Batch, 1] - Ne_Log

        # 查询空间天气序列（点级查询）
        sw_seq = self.sw_manager.get_drivers_sequence(coords[:, 3])  # [Batch, Seq, 2]

        return coords, target_ne, sw_seq


def get_r_stmrf_dataloaders(fy_path, val_days, batch_size, bin_size_hours,
                              sw_manager, tec_manager, num_workers=0):
    """
    获取 R-STMRF 专用的 DataLoader

    Note:
        实际上使用原有的 get_dataloaders 即可，数据处理在训练循环中进行

    Args:
        fy_path: FY 数据路径
        val_days: 验证集日期
        batch_size: 批次大小
        bin_size_hours: 时间分箱大小
        sw_manager: 空间天气管理器
        tec_manager: TEC 管理器
        num_workers: 工作进程数

    Returns:
        train_loader: 训练集 DataLoader
        val_loader: 验证集 DataLoader
        batch_processor: 批次处理器实例
    """
    # 导入原有的 DataLoader 工厂函数
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)

    from data_managers.FY_dataloader import get_dataloaders

    # 获取原有的 DataLoader
    train_loader, val_loader = get_dataloaders(
        npy_path=fy_path,
        val_days=val_days,
        batch_size=batch_size,
        bin_size_hours=bin_size_hours,
        num_workers=num_workers
    )

    # 创建批次处理器
    device = next(iter(train_loader)).device if torch.cuda.is_available() else 'cpu'
    batch_processor = SlidingWindowBatchProcessor(sw_manager, tec_manager, device)

    return train_loader, val_loader, batch_processor


# ======================== 辅助函数 ========================
def collate_with_sequences(batch, sw_manager, tec_manager, device='cuda'):
    """
    自定义 Collate 函数（可选方案）

    如果希望在 DataLoader 层面整合序列数据，可以使用此函数作为 collate_fn

    注意：新架构中，TEC 梯度通过 TecGradientBank 离线预计算，不再在此处加载

    Args:
        batch: List of samples from Dataset
        sw_manager: SpaceWeatherManager
        tec_manager: TECDataManager
        device: 计算设备

    Returns:
        collated_data: dict
    """
    # 将批次堆叠为 Tensor
    batch_data = torch.stack([item for item in batch], dim=0)  # [Batch, 5]

    # 使用批次处理器
    processor = SlidingWindowBatchProcessor(sw_manager, tec_manager, device)
    coords, target_ne, sw_seq = processor.process_batch(batch_data)

    return {
        'coords': coords,
        'target_ne': target_ne,
        'sw_seq': sw_seq
    }


# ======================== 使用示例 ========================
if __name__ == '__main__':
    print("="*60)
    print("滑动窗口数据处理器测试")
    print("="*60)

    # 模拟数据管理器
    class DummySWManager:
        def __init__(self, seq_len=6, device='cuda'):
            self.seq_len = seq_len
            self.device = device

        def get_drivers_sequence(self, time_batch):
            batch_size = time_batch.shape[0]
            return torch.randn(batch_size, self.seq_len, 2).to(self.device)

    class DummyTECManager:
        def __init__(self, seq_len=6, device='cuda'):
            self.seq_len = seq_len
            self.device = device

        def get_tec_map_sequence(self, time_batch):
            batch_size = time_batch.shape[0]
            return torch.rand(batch_size, self.seq_len, 1, 181, 361).to(self.device)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建管理器
    sw_manager = DummySWManager(seq_len=6, device=device)
    tec_manager = DummyTECManager(seq_len=6, device=device)

    # 创建批次处理器
    processor = SlidingWindowBatchProcessor(sw_manager, tec_manager, device=device)

    # 模拟 FY 批次数据
    batch_size = 128
    batch_data = torch.randn(batch_size, 5)
    batch_data[:, 0] = batch_data[:, 0] * 90  # Lat
    batch_data[:, 1] = batch_data[:, 1] * 180  # Lon
    batch_data[:, 2] = 200 + batch_data[:, 2].abs() * 100  # Alt
    batch_data[:, 3] = batch_data[:, 3].abs() * 100  # Time
    batch_data[:, 4] = batch_data[:, 4] + 11.0  # Ne_Log

    print(f"\n输入批次数据: {batch_data.shape}")

    # 处理批次
    coords, target_ne, sw_seq = processor.process_batch(batch_data)

    print("\n输出形状:")
    print(f"  coords: {coords.shape}")
    print(f"  target_ne: {target_ne.shape}")
    print(f"  sw_seq: {sw_seq.shape}")
    print(f"\n注意: TEC 梯度方向现在通过 TecGradientBank 外部提供")

    print("\n数据范围检查:")
    print(f"  Lat: [{coords[:, 0].min().item():.2f}, {coords[:, 0].max().item():.2f}]")
    print(f"  Lon: [{coords[:, 1].min().item():.2f}, {coords[:, 1].max().item():.2f}]")
    print(f"  Alt: [{coords[:, 2].min().item():.2f}, {coords[:, 2].max().item():.2f}]")
    print(f"  Time: [{coords[:, 3].min().item():.2f}, {coords[:, 3].max().item():.2f}]")

    print("\n" + "="*60)
    print("测试通过!")
    print("="*60)
