"""
预计算 TEC 梯度方向库（离线处理）

功能：
    1. 加载原始 TEC 数据 (720, 71, 73)
    2. 应用与 TECDataManager 相同的填充 → (720, 73, 73)
    3. 使用 Sobel 滤波器计算水平梯度 (grad_lat, grad_lon)
    4. 归一化为单位向量（消除幅值依赖，仅保留方向）
    5. 保存为 float16 以节省磁盘空间

输出：
    tec_gradient_bank.npy: (720, 2, 73, 73) dtype=float16
        - Channel 0: normalized grad_lat
        - Channel 1: normalized grad_lon

使用场景：
    替代 ConvLSTM 在线计算，训练时使用 tec_gradient_bank.py 进行快速查询

作者：R-STMRF Team
日期：2026-01-30
"""

import numpy as np
import argparse
import os
from scipy.ndimage import sobel
import sys


def pad_tec_data(raw_data):
    """
    应用与 TECDataManager 相同的纬度填充

    原始: (720, 71, 73) - Lat: -87.5 ~ 87.5 (步长 2.5°)
    填充: (720, 73, 73) - Lat: -90 ~ 90 (前后各填充 1 个点，复制边界)

    Args:
        raw_data: (T, 71, 73) 原始 TEC 数据

    Returns:
        (T, 73, 73) 填充后的 TEC 数据
    """
    # 纬度维度 (axis=1) 前后各填充 1 个点，使用 'edge' 模式（复制边界）
    padded_data = np.pad(raw_data, ((0, 0), (1, 1), (0, 0)), mode='edge')
    return padded_data


def compute_sobel_gradients(tec_maps):
    """
    使用 Sobel 滤波器计算水平梯度（稳健于噪声）

    Sobel 滤波器同时进行平滑和微分，适合提取 TEC 的全局梯度趋势

    Args:
        tec_maps: (T, H, W) TEC 地图序列

    Returns:
        grad_lat: (T, H, W) 纬度方向梯度（∂TEC/∂lat）
        grad_lon: (T, H, W) 经度方向梯度（∂TEC/∂lon）
    """
    T, H, W = tec_maps.shape
    grad_lat = np.zeros_like(tec_maps)
    grad_lon = np.zeros_like(tec_maps)

    print(f"  计算 Sobel 梯度（共 {T} 个时间步）...")

    for t in range(T):
        # Sobel 滤波器: axis=0 对应纬度（行），axis=1 对应经度（列）
        grad_lat[t] = sobel(tec_maps[t], axis=0)  # ∂TEC/∂lat
        grad_lon[t] = sobel(tec_maps[t], axis=1)  # ∂TEC/∂lon

        if (t + 1) % 100 == 0:
            print(f"    进度: {t+1}/{T}")

    return grad_lat, grad_lon


def normalize_to_unit_vectors(grad_lat, grad_lon, eps=1e-8):
    """
    将梯度归一化为单位向量（仅保留方向，消除幅值）

    归一化公式:
        norm = sqrt(grad_lat^2 + grad_lon^2)
        unit_grad_lat = grad_lat / (norm + eps)
        unit_grad_lon = grad_lon / (norm + eps)

    物理意义:
        - 单位向量表示 TEC 水平梯度的方向
        - 损失函数使用余弦相似度，需要单位向量
        - eps 避免除零（平坦区域）

    Args:
        grad_lat: (T, H, W) 纬度梯度
        grad_lon: (T, H, W) 经度梯度
        eps: 数值稳定项

    Returns:
        unit_grad_lat: (T, H, W) 归一化纬度梯度
        unit_grad_lon: (T, H, W) 归一化经度梯度
    """
    # 计算 L2 范数
    norm = np.sqrt(grad_lat**2 + grad_lon**2) + eps

    # 归一化
    unit_grad_lat = grad_lat / norm
    unit_grad_lon = grad_lon / norm

    # 统计信息（用于验证）
    mean_norm = np.mean(np.sqrt(unit_grad_lat**2 + unit_grad_lon**2))
    print(f"  归一化后的平均向量范数: {mean_norm:.6f} (应接近 1.0)")

    return unit_grad_lat, unit_grad_lon


def main(tec_path, output_path, dtype='float16'):
    """
    主处理流程

    Args:
        tec_path: 输入 TEC 数据路径 (*.npy)
        output_path: 输出梯度库路径 (*.npy)
        dtype: 保存数据类型 ('float16' 或 'float32')
    """
    print("="*70)
    print("TEC 梯度方向库预计算")
    print("="*70)

    # 1. 加载原始 TEC 数据
    print(f"\n[步骤 1/5] 加载 TEC 数据: {tec_path}")
    if not os.path.exists(tec_path):
        raise FileNotFoundError(f"TEC 文件未找到: {tec_path}")

    raw_data = np.load(tec_path).astype(np.float32)
    print(f"  原始形状: {raw_data.shape}")
    print(f"  数据类型: {raw_data.dtype}")
    print(f"  数值范围: [{raw_data.min():.2f}, {raw_data.max():.2f}]")

    # 验证形状
    if raw_data.ndim != 3:
        raise ValueError(f"期望 3D 数据 (T, Lat, Lon)，实际形状: {raw_data.shape}")

    T, orig_lat, orig_lon = raw_data.shape
    print(f"  时间步: {T}, 原始纬度点: {orig_lat}, 经度点: {orig_lon}")

    # 2. 应用纬度填充
    print(f"\n[步骤 2/5] 纬度填充（与 TECDataManager 一致）")
    padded_data = pad_tec_data(raw_data)
    print(f"  填充后形状: {padded_data.shape}")

    # 3. 计算 Sobel 梯度
    print(f"\n[步骤 3/5] 计算 Sobel 梯度（稳健于噪声）")
    grad_lat, grad_lon = compute_sobel_gradients(padded_data)
    print(f"  梯度形状: grad_lat={grad_lat.shape}, grad_lon={grad_lon.shape}")
    print(f"  grad_lat 范围: [{grad_lat.min():.2f}, {grad_lat.max():.2f}]")
    print(f"  grad_lon 范围: [{grad_lon.min():.2f}, {grad_lon.max():.2f}]")

    # 4. 归一化为单位向量
    print(f"\n[步骤 4/5] 归一化为单位向量（仅保留方向）")
    unit_grad_lat, unit_grad_lon = normalize_to_unit_vectors(grad_lat, grad_lon)

    # 5. 保存为压缩格式
    print(f"\n[步骤 5/5] 保存梯度库: {output_path}")

    # 堆叠为 (T, 2, H, W) 格式
    gradient_bank = np.stack([unit_grad_lat, unit_grad_lon], axis=1)
    print(f"  输出形状: {gradient_bank.shape}")

    # 转换数据类型
    if dtype == 'float16':
        gradient_bank = gradient_bank.astype(np.float16)
        print(f"  数据类型: float16 (节省磁盘空间)")
    else:
        gradient_bank = gradient_bank.astype(np.float32)
        print(f"  数据类型: float32")

    # 保存
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    np.save(output_path, gradient_bank)

    # 文件大小统计
    file_size_mb = os.path.getsize(output_path) / (1024**2)
    print(f"  文件大小: {file_size_mb:.2f} MB")

    # 验证保存
    print(f"\n[验证] 重新加载验证...")
    loaded = np.load(output_path)
    print(f"  重载形状: {loaded.shape}")
    print(f"  重载类型: {loaded.dtype}")
    print(f"  数值范围: [{loaded.min():.4f}, {loaded.max():.4f}]")

    # 计算一些统计量验证归一化
    sample_norms = np.sqrt(loaded[:10, 0, :, :]**2 + loaded[:10, 1, :, :]**2)
    print(f"  前10帧的平均向量范数: {sample_norms.mean():.6f} (应接近 1.0)")

    print("\n" + "="*70)
    print("✓ 预计算完成！")
    print("="*70)
    print(f"\n下一步: 在训练脚本中使用 TecGradientBank 加载此文件")
    print(f"示例:\n  from tec_gradient_bank import TecGradientBank")
    print(f"  gradient_bank = TecGradientBank('{output_path}')")
    print("="*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='预计算 TEC 梯度方向库')
    parser.add_argument('--tec_path', type=str, required=True,
                        help='输入 TEC 数据路径 (*.npy)')
    parser.add_argument('--output_path', type=str, required=True,
                        help='输出梯度库路径 (*.npy)')
    parser.add_argument('--dtype', type=str, default='float16', choices=['float16', 'float32'],
                        help='保存数据类型 (默认: float16)')

    args = parser.parse_args()

    try:
        main(args.tec_path, args.output_path, args.dtype)
    except Exception as e:
        print(f"\n❌ 错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
