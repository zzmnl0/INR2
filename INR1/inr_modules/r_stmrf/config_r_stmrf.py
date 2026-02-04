"""
R-STMRF 模型配置文件

包含所有训练和模型的全局配置参数

性能优化：
- 自动检测CUDA并启用GPU加速优化
- 多进程数据加载
- 混合精度训练（AMP）
- DataLoader预取
"""

import torch
import os

# ==================== 自动检测CUDA并设置优化参数 ====================
_CUDA_AVAILABLE = torch.cuda.is_available()
_DEVICE = 'cuda' if _CUDA_AVAILABLE else 'cpu'

if _CUDA_AVAILABLE:
    print("  ✓ GPU optimizations enabled")
    # GPU优化配置
    _BATCH_SIZE = 4096  # 增大batch提高GPU利用率
    _NUM_WORKERS = 4  # 多进程数据加载
    _USE_AMP = True  # 启用混合精度训练
    _PIN_MEMORY = True  # 启用pin memory加速CPU-GPU传输
    _PREFETCH_FACTOR = 2  # 预取2个batch
    _PERSISTENT_WORKERS = True  # 保持worker进程

else:
    print("  ✓ CPU optimizations enabled")
    # CPU配置
    _BATCH_SIZE = 2048
    _NUM_WORKERS = 2  # CPU也启用多进程
    _USE_AMP = False  # CPU不支持AMP
    _PIN_MEMORY = False
    _PREFETCH_FACTOR = 2
    _PERSISTENT_WORKERS = False


CONFIG_R_STMRF = {
    # ==================== 数据路径 ====================
    'fy_path': r'D:\FYsatellite\EDP_data\fy_202409_clean.npy',
    'iri_proxy_path': r"D:\code11\IRI01\output_results\iri_september_full_proxy.pth",
    'sw_path': r'D:\FYsatellite\EDP_data\kp\OMNI_Kp_F107_20240901_20241001.txt',
    'tec_path': r'D:\IGS\VTEC\tec_map_data.npy',  # 原始 TEC 数据（用于预计算）
    'gradient_bank_path': r'D:\IGS\VTEC\tec_gradient_bank.npy',  # 预计算的 TEC 梯度库（新架构）
    'save_dir': './checkpoints_r_stmrf',

    # ==================== 数据规格 ====================
    'total_hours': 720.0,  # 总时长（小时）
    'start_date_str': '2024-09-01 00:00:00',  # 起始时间
    'time_res': 0.5,  # IRI 时间分辨率（小时）
    'bin_size_hours': 0.5,  # 时间分箱大小（小时）

    # ==================== 物理参数 ====================
    'lat_range': (-90.0, 90.0),  # 纬度范围
    'lon_range': (-180.0, 180.0),  # 经度范围（注意：-180 与 180 重复）
    'alt_range': (120.0, 500.0),  # 高度范围（km）

    # ==================== 时序学习参数 ====================
    'seq_len': 6,  # 历史窗口长度（时间步）

    # ==================== SIREN 架构参数 ====================
    'basis_dim': 64,  # 空间/时间基函数维度
    'siren_hidden': 128,  # SIREN 隐层维度
    'siren_layers': 3,  # SIREN 隐层数量
    'omega_0': 30.0,  # SIREN 频率因子

    # ==================== 循环网络参数 ====================
    # LSTM (全局环境编码器 - 仍然保留)
    'env_hidden_dim': 64,  # LSTM 隐层维度
    'lstm_layers': 2,  # LSTM 层数
    'lstm_dropout': 0.1,  # LSTM Dropout

    # TEC 相关参数（新架构：离线预计算）
    # - ConvLSTM 已移除，使用 TecGradientBank 替代
    # - TEC 梯度方向通过 precompute_tec_gradient_bank.py 离线计算
    # - 训练时使用 memory-mapped loading + 时间插值

    # ==================== 训练超参数 ====================
    'batch_size': _BATCH_SIZE,  # 批次大小（自动调整：GPU=4096, CPU=2048）
    'lr': 3e-4,  # 学习率
    'weight_decay': 1e-4,  # 权重衰减
    'epochs': 6,  # 训练轮数
    'seed': 42,
    'device': _DEVICE,  # 自动检测 CUDA
    'num_workers': _NUM_WORKERS,  # 多进程数据加载（自动调整：GPU=4, CPU=2）
    'pin_memory': _PIN_MEMORY,  # Pin memory加速CPU-GPU传输
    'prefetch_factor': _PREFETCH_FACTOR,  # 预取batch数量
    'persistent_workers': _PERSISTENT_WORKERS,  # 保持worker进程（减少启动开销）
    'use_memmap': False,  # 是否使用内存映射按需加载数据（节省内存，略慢）
                          # True: Memory-mapped loading (低内存占用，适合大数据集)
                          # False: 全量加载到内存 (高性能，需要足够内存)

    # ==================== 学习率调度 ====================
    'scheduler_type': 'cosine',  # 学习率调度器类型 ['cosine', 'step', 'plateau']
    'warmup_epochs': 3,  # 预热轮数
    'min_lr': 1e-6,  # 最小学习率

    # ==================== 数据划分 ====================
    'val_days': [],  # 验证集日期（留空表示使用随机划分）
    'val_ratio': 0.1,  # 验证集比例（随机划分）

    # ==================== 损失函数权重 ====================
    # 主损失
    'w_mse': 1.0,  # MSE 损失权重（或 Huber Loss）

    # 物理约束损失（v2.0+ 架构）
    'w_chapman': 0.1,  # Chapman 垂直平滑损失权重
    'w_tec_direction': 0.03,  # TEC 梯度方向一致性权重（取较小值避免过约束）
    'physics_loss_freq': 5,  # 物理损失计算频率（每N个batch计算一次，加速训练）
                               # 设为1表示每个batch都计算（无加速）
                               # 设为10表示每10个batch计算一次（推荐，2-3倍加速）

    # ==================== 不确定性学习 ====================
    'use_uncertainty': True,  # 是否启用异方差损失
    'uncertainty_weight': 0.5,  # 不确定性项权重
    'uncertainty_warmup_epochs': 5,  # 前 N 个 epoch 关闭不确定性损失，只用 MSE+物理损失
                                      # 先让模型学会"预测准确"，再学会"预测方差"
    'log_var_min': -10.0,  # log_var 下界（防止方差过小导致 NaN）
    'log_var_max': 10.0,   # log_var 上界（防止方差过大导致数值溢出）
    'log_var_regularization': 0.001,  # log_var 正则化权重（惩罚极端方差，鼓励接近 1）

    # ==================== 模型保存 ====================
    'save_interval': 3,  # 每隔多少个 epoch 保存一次模型
    'save_best_only': True,  # 是否只保存最佳模型

    # ==================== 可视化 ====================
    'plot_interval': 3,  # 每隔多少个 epoch 绘制可视化
    'plot_days': [5, 15, 25],  # 绘制哪些天的高度剖面
    'plot_hours': [0.0, 6.0, 12.0, 18.0],  # 绘制哪些时刻

    # ==================== 早停 ====================
    'early_stopping': True,  # 是否启用早停
    'patience': 5,  # 早停耐心值（验证损失不下降的轮数）

    # ==================== 梯度裁剪 ====================
    'grad_clip': 1.0,  # 梯度裁剪阈值（设为 None 则不裁剪）

    # ==================== 混合精度训练 ====================
    'use_amp': _USE_AMP,  # 自动混合精度（GPU自动启用，CPU禁用）

    # ==================== TEC 梯度对齐参数 ====================
    'tec_gradient_threshold_percentile': 50.0,  # TEC 梯度显著性阈值（百分位数）
}


def get_config_r_stmrf():
    """获取 R-STMRF 配置字典"""
    # 确保保存目录存在
    os.makedirs(CONFIG_R_STMRF['save_dir'], exist_ok=True)
    return CONFIG_R_STMRF


def update_config_r_stmrf(**kwargs):
    """更新配置参数"""
    CONFIG_R_STMRF.update(kwargs)


def print_config_r_stmrf():
    """打印当前配置"""
    print("\n" + "="*7)
    print("R-STMRF 配置参数")
    print("="*7)

    # 分类打印
    categories = {
        '数据路径': ['fy_path', 'iri_proxy_path', 'sw_path', 'tec_path', 'gradient_bank_path', 'save_dir'],
        '数据规格': ['total_hours', 'start_date_str', 'time_res', 'bin_size_hours'],
        '物理参数': ['lat_range', 'lon_range', 'alt_range'],
        '时序参数': ['seq_len'],
        'SIREN 架构': ['basis_dim', 'siren_hidden', 'siren_layers', 'omega_0'],
        '循环网络': ['env_hidden_dim', 'lstm_layers', 'lstm_dropout'],
        '训练超参数': ['batch_size', 'lr', 'weight_decay', 'epochs', 'device', 'num_workers',
                        'pin_memory', 'prefetch_factor', 'persistent_workers', 'use_memmap'],
        '学习率调度': ['scheduler_type', 'warmup_epochs', 'min_lr'],
        '数据划分': ['val_days', 'val_ratio'],
        '损失权重': ['w_mse', 'w_chapman', 'w_tec_direction', 'physics_loss_freq'],
        '不确定性学习': ['use_uncertainty', 'uncertainty_weight', 'uncertainty_warmup_epochs',
                        'log_var_min', 'log_var_max', 'log_var_regularization'],
        '其他': ['save_interval', 'save_best_only', 'plot_interval', 'early_stopping',
                 'patience', 'grad_clip', 'use_amp'],
    }

    for category, keys in categories.items():
        print(f"\n【{category}】")
        for key in keys:
            if key in CONFIG_R_STMRF:
                value = CONFIG_R_STMRF[key]
                print(f"  {key:30s}: {value}")

    print("\n" + "="*7 + "\n")


def validate_config():
    """验证配置合理性"""
    config = CONFIG_R_STMRF

    # 检查必要路径
    required_paths = ['fy_path', 'iri_proxy_path', 'sw_path', 'tec_path']
    for path_key in required_paths:
        path = config[path_key]
        if not os.path.exists(path):
            print(f"警告: {path_key} 路径不存在: {path}")

    # 检查参数合理性
    assert config['seq_len'] > 0, "seq_len 必须大于 0"
    assert config['basis_dim'] > 0, "basis_dim 必须大于 0"
    assert config['batch_size'] > 0, "batch_size 必须大于 0"
    assert 0 < config['lr'] < 1, "学习率必须在 (0, 1) 之间"

    print("✅ 配置验证通过")


# ======================== 使用示例 ========================
if __name__ == '__main__':
    print_config_r_stmrf()
    validate_config()
