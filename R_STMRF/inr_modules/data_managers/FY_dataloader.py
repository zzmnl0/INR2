

import torch
from torch.utils.data import Dataset, Sampler, DataLoader
import numpy as np
import os
from typing import List, Iterator, Tuple

class FY3D_Dataset(Dataset):
    """
    FY3D 卫星电离层数据 Dataset。

    [Time-Aware Strategy Step 1]:
    在初始化时，根据 'bin_size_hours' 将数据按时间分组。
    这建立了一个 'Bin ID -> Sample Indices' 的映射表。

    [Memory-Efficient Loading]:
    支持两种加载模式：
    - use_memmap=False: 一次性加载到内存（原始行为，速度快但内存占用大）
    - use_memmap=True: 使用内存映射按需加载（内存友好，速度略慢）
    """
    def __init__(self, npy_path: str, mode: str = 'train', val_days: List[int] = None,
                 bin_size_hours: float = 3.0, use_memmap: bool = False):
        super().__init__()

        if val_days is None:
            val_days = []

        self.use_memmap = use_memmap
        self.npy_path = npy_path

        print(f"Loading data from {npy_path}...")
        print(f"  内存模式: {'Memory-Mapped (按需加载)' if use_memmap else '全量加载到内存'}")

        try:
            if use_memmap:
                # 使用memmap按需加载（节省内存）
                raw_data = np.load(npy_path, mmap_mode='r')
                print(f"  ✓ Memory-mapped加载成功，形状: {raw_data.shape}")
            else:
                # 全量加载到内存（原始行为）
                raw_data = np.load(npy_path).astype(np.float32)
        except FileNotFoundError:
            workspace_file = os.path.join(os.getcwd(), os.path.basename(npy_path))
            if os.path.exists(workspace_file):
                print(f"Warning: {npy_path} not found. Using {workspace_file} instead.")
                if use_memmap:
                    raw_data = np.load(workspace_file, mmap_mode='r')
                else:
                    raw_data = np.load(workspace_file).astype(np.float32)
            else:
                raise FileNotFoundError(f"Could not find file at {npy_path} or {workspace_file}")

        # --- Filter NaNs ---
        # 注意：memmap模式下，需要先创建索引再过滤
        print(f"  检查NaN值...")
        isnan_mask = np.isnan(raw_data).any(axis=1)
        if isnan_mask.any():
            print(f"  Warning: Found {np.sum(isnan_mask)} samples with NaN values. Filtering them out...")
            valid_indices = np.where(~isnan_mask)[0]
            if use_memmap:
                # memmap模式：只存储有效索引，不复制数据
                self.data = raw_data
                self.valid_indices = valid_indices
            else:
                # 全量模式：复制有效数据
                self.data = raw_data[~isnan_mask]
                self.valid_indices = None
        else:
            self.data = raw_data
            self.valid_indices = None if not use_memmap else np.arange(len(raw_data))

        # 获取工作索引（考虑NaN过滤）
        if self.valid_indices is not None:
            working_indices = self.valid_indices
            working_data = self.data[working_indices]
        else:
            working_indices = np.arange(len(self.data))
            working_data = self.data

        # --- Sanity Check: 验证时间格式 ---
        print(f"  验证时间格式...")
        relative_hours_check = working_data[:, 3]
        max_hour = np.max(relative_hours_check)
        if max_hour <= 48:
            raise ValueError(f"Input data appears to use Daily Hours (0-24). Max hour: {max_hour:.2f}. Expected Continuous Hours (0-720).")

        # --- 划分 Train/Val ---
        print(f"  划分训练/验证集 (mode={mode})...")
        relative_hours = working_data[:, 3]
        days = np.floor(relative_hours / 24.0).astype(int)
        is_val = np.isin(days, val_days)

        if mode == 'val':
            self.selected_indices = working_indices[is_val]
        else:
            self.selected_indices = working_indices[~is_val]

        print(f"Mode '{mode}': {len(self.selected_indices)} samples selected.")

        # --- [关键步骤] 预计算时间分箱 (Bin Indexing) ---
        print(f"  计算时间分箱 (bin_size={bin_size_hours}h)...")
        # 1. 获取过滤后数据的 relative_hours
        filtered_hours = self.data[self.selected_indices, 3]

        # 2. 计算 Bin ID (例如 3小时一个Bin, 0-3h -> Bin 0)
        # 这保证了同一个 Bin 内的数据对应的 IRI 背景场帧索引是相同的 (Frame t, Frame t+1)
        self.bin_ids = np.floor(filtered_hours / bin_size_hours).astype(int)

        # 3. 构建高效索引表: {bin_id: [indices...]}
        # 使用 argsort 避免循环，极大加速初始化
        sorted_indices = np.argsort(self.bin_ids)
        sorted_bin_ids = self.bin_ids[sorted_indices]
        unique_bins, split_points = np.unique(sorted_bin_ids, return_index=True)
        grouped_indices = np.split(sorted_indices, split_points[1:])

        self.indices_by_bin = {}
        for bin_id, indices in zip(unique_bins, grouped_indices):
            self.indices_by_bin[bin_id] = indices

        print(f"  ✓ Data binned into {len(self.indices_by_bin)} time bins")
        if use_memmap:
            print(f"  ✓ Memory-mapped模式：数据将按需从磁盘读取，大幅节省内存")

    def __len__(self):
        return len(self.selected_indices)

    def __getitem__(self, idx):
        # 映射到实际数据索引
        actual_idx = self.selected_indices[idx]

        # 按需读取数据
        if self.use_memmap:
            # memmap模式：从磁盘读取单个样本（节省内存）
            data_sample = self.data[actual_idx].astype(np.float32)
        else:
            # 全量模式：从内存读取
            data_sample = self.data[idx]

        return torch.from_numpy(data_sample)


class TimeBinSampler(Sampler):
    """
    [Time-Aware Strategy Step 2]: 自定义 Sampler
    
    执行逻辑:
    1. 每次迭代时，随机打乱 Bin 的顺序 (Bin-Level Shuffle)。
    2. 锁定一个 Bin 后，取出该 Bin 所有数据，并在内部打乱 (Intra-Bin Shuffle)。
    3. 按照 batch_size 切分该 Bin 的数据。
    
    结果: 
    yield 出的每一个 batch_indices 列表，其所有索引都严格属于同一个 Time Bin。
    """
    def __init__(self, dataset: FY3D_Dataset, batch_size: int, shuffle: bool = True, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
    def __iter__(self) -> Iterator[List[int]]:
        # 获取所有存在的 Bin ID
        bin_ids = list(self.dataset.indices_by_bin.keys())
        
        # 1. Bin Level Shuffle: 随机决定先学哪个时间段 (例如先学 Day 5, 再学 Day 1)
        if self.shuffle:
            np.random.shuffle(bin_ids)
            
        for bin_id in bin_ids:
            # 获取该 Bin 下的所有样本索引
            # 注意: 这里使用 copy() 确保不影响原始索引顺序，虽然在当前逻辑下非必要，但为了安全
            indices = self.dataset.indices_by_bin[bin_id].copy()
            
            # 2. Batch Level Shuffle: 打乱该时间段内的空间点顺序
            if self.shuffle:
                np.random.shuffle(indices)
                
            # 3. 生成 Batches (严格限制在当前 Bin 内)
            num_samples = len(indices)
            for i in range(0, num_samples, self.batch_size):
                batch_indices = indices[i : i + self.batch_size]
                
                # 处理最后一个不足 batch_size 的包
                if len(batch_indices) < self.batch_size and self.drop_last:
                    continue
                    
                # yield 给 DataLoader
                yield batch_indices.tolist()

    def __len__(self):
        # 准确计算 Batch 总数
        count = 0
        for indices in self.dataset.indices_by_bin.values():
            n = len(indices)
            if self.drop_last:
                count += n // self.batch_size
            else:
                count += (n + self.batch_size - 1) // self.batch_size
        return count


def get_dataloaders(
    npy_path: str,
    val_days: List[int],
    batch_size: int = 1024,
    bin_size_hours: float = 3.0,
    num_workers: int = 0,
    use_memmap: bool = False
):
    """
    工厂函数: 组装 Dataset 和 Time-Aware Sampler

    Args:
        npy_path: 数据文件路径
        val_days: 验证集日期列表
        batch_size: 批次大小
        bin_size_hours: 时间分箱大小（小时）
        num_workers: DataLoader工作进程数
        use_memmap: 是否使用内存映射按需加载（节省内存）

    Returns:
        train_loader, val_loader
    """
    train_dataset = FY3D_Dataset(npy_path, mode='train', val_days=val_days,
                                  bin_size_hours=bin_size_hours, use_memmap=use_memmap)
    val_dataset = FY3D_Dataset(npy_path, mode='val', val_days=val_days,
                                bin_size_hours=bin_size_hours, use_memmap=use_memmap)
    
    # 训练集开启 Shuffle (Time-Aware Shuffle)
    train_sampler = TimeBinSampler(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    # 验证集关闭 Shuffle (顺序评估)
    val_sampler = TimeBinSampler(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    # DataLoader 必须使用 batch_sampler 参数
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=num_workers, pin_memory=False)
    
    return train_loader, val_loader

# (测试代码块保持不变，省略)


if __name__ == "__main__":
    # --- 配置 ---
    NPY_FILE_PATH = r"D:\FYsatellite\EDP_data\fy_202409_clean.npy"
    VAL_DAYS = [5, 15, 25] # 假设这些天作为验证集
    BATCH_SIZE = 512       # CPU: Reduced batch size (was 4096)
    BIN_SIZE = 3.0         # 3小时窗口
    
    print("=== 初始化 DataLoader 管道 ===")
    
    # 检查文件是否存在，如果不存在创建一个假的用于测试（仅用于演示代码可运行性）
    if not os.path.exists(NPY_FILE_PATH):
        print(f"文件 {NPY_FILE_PATH} 未找到。")
        # 尝试在本地生成一个用于演示的 dummy 文件
        print("正在生成 Dummy 数据用于测试...")
        dummy_data = np.zeros((100000, 5), dtype=np.float32)
        # Lat, Lon, Alt
        dummy_data[:, 0] = np.random.uniform(-90, 90, 100000)
        dummy_data[:, 1] = np.random.uniform(-180, 180, 100000)
        dummy_data[:, 2] = np.random.uniform(100, 800, 100000)
        # Relative Hour (0 to 720 hours, ~30 days)
        dummy_data[:, 3] = np.random.uniform(0, 30*24, 100000) 
        # Ne_Log
        dummy_data[:, 4] = np.random.uniform(8, 13, 100000)
        
        # 保存到当前目录
        NPY_FILE_PATH = "dummy_fy_data.npy"
        np.save(NPY_FILE_PATH, dummy_data)
        print(f"Dummy data saved to {NPY_FILE_PATH}")

    # 获取 Loaders
    train_loader, val_loader = get_dataloaders(
        npy_path=NPY_FILE_PATH,
        val_days=VAL_DAYS,
        batch_size=BATCH_SIZE,
        bin_size_hours=BIN_SIZE
    )
    
    print("\n=== 验证 Train Loader ===")
    print(f"Train batches: {len(train_loader)}")
    
    # 迭代一个 Batch 验证逻辑
    for batch_idx, batch_data in enumerate(train_loader):
        # batch_data shape: (Batch_Size, 5)
        relative_hours = batch_data[:, 3]
        
        min_h = relative_hours.min().item()
        max_h = relative_hours.max().item()
        
        # 验证是否在同一个 BIN 内 (跨度 < BIN_SIZE)
        # 注意: 严格来说，是 floor(h/3) 必须相同
        bin_ids = torch.floor(relative_hours / BIN_SIZE).int()
        unique_bins = torch.unique(bin_ids)
        
        if len(unique_bins) != 1:
            print(f"Error: Batch {batch_idx} contains data from multiple bins: {unique_bins.tolist()}")
            break
            
        print(f"Batch {batch_idx} check passed:")
        print(f"  Shape: {batch_data.shape}")
        print(f"  Bin ID: {unique_bins.item()}")
        print(f"  Hour Range: [{min_h:.2f}, {max_h:.2f}] (Span: {max_h - min_h:.4f}h)")
        
        # 只打印第一个就退出
        break

    print("\n=== 验证 Val Loader ===")
    print(f"Val batches: {len(val_loader)}")
    # 简单迭代一下
    for i, batch in enumerate(val_loader):
        pass
    print("Val iteration complete.")

