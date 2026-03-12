"""
R-STMRF 物理约束损失函数

物理约束损失：
    1. Chapman 垂直平滑损失（垂直方向二阶导数约束）
    2. TEC 梯度方向一致性损失（v2.0+ - 仅约束方向，不约束幅值）

设计理念：
    - TEC 仅作为"时序水平梯度方向约束"
    - 不约束 TEC 幅值，不要求 Ne 积分等于 TEC
    - 仅要求 Ne_fused 的水平变化方向不与 TEC 主导方向相矛盾
    - 适用于高度覆盖不完整的情况，保持物理一致性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def chapman_smoothness_loss(pred_ne, coords, alt_idx=2, weight_second_order=1.0):
    """
    Chapman 形态约束：垂直平滑损失

    约束电子密度在高度方向的二阶导数，确保符合 Chapman 层的平滑特征，
    避免非物理震荡。

    Chapman 函数的特点：
        - 峰值平滑
        - 垂直剖面无震荡
        - 负曲率（凹函数）

    Args:
        pred_ne: 预测 Ne 值 [Batch, 1]
        coords: 坐标 [Batch, 4] (Lat, Lon, Alt, Time)，requires_grad=True
        alt_idx: 高度维度索引（默认 2）
        weight_second_order: 二阶导数惩罚权重

    Returns:
        scalar loss

    Note:
        二阶导数计算在AMP下可能导致数值不稳定，此函数强制使用float32
    """
    # 确保 coords 需要梯度
    if not coords.requires_grad:
        raise ValueError("coords 必须设置 requires_grad=True")

    # ⚠️ 关键修复：禁用AMP以避免二阶导数计算时的数值溢出
    # 二阶导数对精度非常敏感，必须使用float32
    with torch.amp.autocast('cuda', enabled=False):
        # 确保所有输入都是float32
        pred_ne_fp32 = pred_ne.float()
        coords_fp32 = coords.float()

        # 1. 计算一阶导数 ∂Ne/∂coords
        grad_first = torch.autograd.grad(
            outputs=pred_ne_fp32,
            inputs=coords_fp32,
            grad_outputs=torch.ones_like(pred_ne_fp32),
            create_graph=True,  # 必须保留计算图以计算二阶导数
            retain_graph=True,
            only_inputs=True
        )[0]  # [Batch, 4]

        # 2. 提取高度方向的一阶导数
        grad_alt = grad_first[:, alt_idx]  # [Batch]

        # 3. 计算二阶导数 ∂²Ne/∂h²
        grad_second_alt = torch.autograd.grad(
            outputs=grad_alt,
            inputs=coords_fp32,
            grad_outputs=torch.ones_like(grad_alt),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0][:, alt_idx]  # [Batch]

        # 4. 惩罚二阶导数的绝对值（或平方）
        # 选项 A: L2 惩罚（平滑）
        loss_second = torch.mean(grad_second_alt ** 2)

        # 选项 B: L1 惩罚（稀疏震荡抑制）
        # loss_second = torch.mean(torch.abs(grad_second_alt))

        # 可选：额外惩罚一阶导数的突变（TV 正则）
        # loss_first_tv = torch.mean(torch.abs(grad_alt[1:] - grad_alt[:-1]))

        total_loss = weight_second_order * loss_second

    return total_loss


def tec_gradient_direction_consistency_loss(pred_ne, coords, tec_grad_direction, coords_normalized):
    """
    TEC 梯度方向一致性损失（新设计 - 仅约束方向，不约束幅值）

    新设计理念：
        - TEC 仅作为"时序水平梯度方向约束"
        - 不约束 TEC 幅值，不要求 Ne 积分等于 TEC
        - 仅约束 Ne_fused 的水平变化方向不与 TEC 主导方向相矛盾
        - 使用余弦相似度约束方向，允许幅值自由变化

    Args:
        pred_ne: 预测 Ne_fused 值 [Batch, 1]（IRI背景 + SIREN残差）
        coords: 坐标 [Batch, 4] (Lat, Lon, Alt, Time)，requires_grad=True
        tec_grad_direction: TEC 梯度方向场 [Batch, 2, H, W] - (grad_lat, grad_lon)
        coords_normalized: 归一化坐标 [Batch, 2] - (lat_n, lon_n) 用于采样

    Returns:
        scalar loss (1 - 平均余弦相似度)
    """
    # 1. 计算 Ne_fused 的水平梯度（使用自动微分）
    grad_ne = torch.autograd.grad(
        outputs=pred_ne,
        inputs=coords,
        grad_outputs=torch.ones_like(pred_ne),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]  # [Batch, 4]

    # 提取水平梯度（lat, lon 方向）
    grad_ne_lat = grad_ne[:, 0]  # [Batch] - ∂Ne/∂lat
    grad_ne_lon = grad_ne[:, 1]  # [Batch] - ∂Ne/∂lon
    grad_ne_horizontal = torch.stack([grad_ne_lat, grad_ne_lon], dim=1)  # [Batch, 2]

    # 2. 从 TEC 梯度方向场中采样期望的梯度方向
    # tec_grad_direction: [Batch, 2, H, W]
    # coords_normalized: [Batch, 2] - (lat_n, lon_n) in [-1, 1]

    lat_n, lon_n = coords_normalized[:, 0], coords_normalized[:, 1]
    batch_size = lat_n.shape[0]

    # 构建 grid_sample 坐标 [Batch, 1, 1, 2]
    grid_coords = torch.stack([lon_n, lat_n], dim=-1)  # [Batch, 2]
    grid_coords = grid_coords.view(batch_size, 1, 1, 2)  # [Batch, 1, 1, 2]

    # 采样 TEC 梯度方向
    tec_grad_expected = F.grid_sample(
        tec_grad_direction, grid_coords,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )  # [Batch, 2, 1, 1]

    tec_grad_expected = tec_grad_expected.squeeze(-1).squeeze(-1)  # [Batch, 2]

    # 3. 计算方向一致性（余弦相似度）
    # 归一化梯度向量（只保留方向，去除幅值）
    grad_ne_norm = F.normalize(grad_ne_horizontal, p=2, dim=1, eps=1e-6)  # [Batch, 2]
    tec_grad_norm = F.normalize(tec_grad_expected, p=2, dim=1, eps=1e-6)  # [Batch, 2]

    # 计算余弦相似度
    cosine_sim = F.cosine_similarity(grad_ne_norm, tec_grad_norm, dim=1)  # [Batch]

    # 损失: 1 - 余弦相似度（最小化时，余弦相似度趋向 1，即方向一致）
    loss = torch.mean(1.0 - cosine_sim)

    return loss


def combined_physics_loss(pred_ne, coords, tec_grad_direction=None, coords_normalized=None,
                           w_chapman=0.1, w_tec_direction=0.05,
                           tec_lat_range=(-90, 90), tec_lon_range=(-180, 180)):
    """
    组合物理约束损失（v2.0+ 架构）

    物理约束：
        1. Chapman 垂直平滑（二阶导数约束）
        2. TEC 梯度方向一致性（仅约束方向，不约束幅值）

    设计原则：
        - TEC 仅作为梯度方向引导，不约束数值
        - 不强制 Ne 积分等于观测 TEC
        - 适用于高度覆盖不完整的场景

    Args:
        pred_ne: 预测 Ne_fused 值 [Batch, 1]（IRI背景 + SIREN残差）
        coords: 坐标 [Batch, 4] (Lat, Lon, Alt, Time), requires_grad=True
        tec_grad_direction: TEC 梯度方向场 [Batch, 2, H, W]
                           通过 TecGradientBank 预计算获得
        coords_normalized: 归一化坐标 [Batch, 2] (lat_n, lon_n)
                          用于从 tec_grad_direction 采样
        w_chapman: Chapman 垂直平滑损失权重
        w_tec_direction: TEC 梯度方向一致性权重（建议: 0.01-0.05）
        tec_lat_range: TEC 数据纬度范围（用于坐标归一化）
        tec_lon_range: TEC 数据经度范围（用于坐标归一化）

    Returns:
        total_loss: scalar，加权总损失
        loss_dict: 各项损失的字典
            - chapman: Chapman 垂直平滑损失
            - tec_direction: TEC 梯度方向一致性损失
            - physics_total: 总物理损失
    """
    # 1. Chapman 垂直平滑
    loss_chapman = chapman_smoothness_loss(pred_ne, coords, alt_idx=2)

    # 2. TEC 梯度方向一致性（v2.0+ 架构）
    if tec_grad_direction is not None and coords_normalized is not None:
        loss_tec_direction = tec_gradient_direction_consistency_loss(
            pred_ne, coords, tec_grad_direction, coords_normalized
        )
    else:
        loss_tec_direction = torch.tensor(0.0, device=pred_ne.device)

    # 总损失
    total_loss = (w_chapman * loss_chapman +
                  w_tec_direction * loss_tec_direction)

    loss_dict = {
        'chapman': loss_chapman.item(),
        'tec_direction': loss_tec_direction.item() if isinstance(loss_tec_direction, torch.Tensor) else 0.0,
        'physics_total': total_loss.item()
    }

    return total_loss, loss_dict


# ======================== 测试代码 ========================
if __name__ == '__main__':
    print("="*60)
    print("R-STMRF 物理损失测试")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 模拟数据
    batch_size = 128
    coords = torch.randn(batch_size, 4, requires_grad=True, device=device)
    coords[:, 0] = coords[:, 0] * 90  # Lat
    coords[:, 1] = coords[:, 1] * 180  # Lon
    coords[:, 2] = 200 + coords[:, 2].abs() * 100  # Alt
    coords[:, 3] = coords[:, 3].abs() * 100  # Time

    # 模拟模型输出
    pred_ne = torch.randn(batch_size, 1, requires_grad=True, device=device) + 11.0

    # 模拟 TEC 梯度方向（预计算）
    tec_grad_direction = torch.randn(batch_size, 2, 73, 73, device=device)
    # 归一化为单位向量
    norm = torch.sqrt((tec_grad_direction ** 2).sum(dim=1, keepdim=True)) + 1e-8
    tec_grad_direction = tec_grad_direction / norm

    # 模拟归一化坐标
    coords_normalized = torch.randn(batch_size, 2, device=device)

    # 测试 Chapman 损失
    print("\n[测试 1] Chapman Smoothness Loss")
    loss_chapman = chapman_smoothness_loss(pred_ne, coords, alt_idx=2)
    print(f"  Chapman Loss: {loss_chapman.item():.6f}")

    # 测试 TEC 梯度方向一致性损失
    print("\n[测试 2] TEC Gradient Direction Consistency Loss")
    loss_tec = tec_gradient_direction_consistency_loss(
        pred_ne, coords, tec_grad_direction, coords_normalized
    )
    print(f"  TEC Direction Loss: {loss_tec.item():.6f}")

    # 测试组合损失
    print("\n[测试 3] Combined Physics Loss")
    total_loss, loss_dict = combined_physics_loss(
        pred_ne, coords,
        tec_grad_direction=tec_grad_direction,
        coords_normalized=coords_normalized,
        w_chapman=0.1,
        w_tec_direction=0.05
    )
    print(f"  Total Physics Loss: {total_loss.item():.6f}")
    for key, val in loss_dict.items():
        print(f"    {key}: {val:.6f}")

    # 测试梯度流动
    print("\n[测试 4] 梯度流动")
    total_loss.backward()
    print(f"  pred_ne 梯度范数: {pred_ne.grad.norm().item():.6f}")
    print(f"  coords 梯度范数: {coords.grad.norm().item():.6f}")

    print("\n" + "="*60)
    print("所有测试通过!")
    print("="*60)
