"""
R-STMRF 训练脚本

完整的训练流程，包括：
    - 数据加载（保留 TimeBinSampler）
    - 模型初始化
    - 物理约束损失
    - 训练和验证循环
    - 模型保存
"""

import os
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm

# 添加模块路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from .config_r_stmrf import get_config_r_stmrf, print_config_r_stmrf
from .r_stmrf_model import R_STMRF_Model
from .physics_losses_r_stmrf import combined_physics_loss
from .sliding_dataset import SlidingWindowBatchProcessor
from .tec_gradient_bank import TecGradientBank

from data_managers import SpaceWeatherManager, TECDataManager, IRINeuralProxy
from data_managers.FY_dataloader import FY3D_Dataset, TimeBinSampler


class SubsetTimeBinSampler(TimeBinSampler):
    """
    TimeBinSampler 的 Subset 版本

    用于在 random_split 之后仍然使用时间分箱策略

    关键修复：将原始数据集的绝对索引转换为 Subset 的相对索引
    """
    def __init__(self, subset: Subset, batch_size: int, shuffle: bool = True, drop_last: bool = False):
        # 获取底层的 FY3D_Dataset
        base_dataset = subset.dataset
        subset_indices = subset.indices

        # 创建从原始索引到 Subset 相对索引的映射
        # 例如：subset.indices = [100, 200, 300] -> {100: 0, 200: 1, 300: 2}
        original_to_subset_idx = {orig_idx: subset_idx
                                  for subset_idx, orig_idx in enumerate(subset_indices)}

        # 构建 Subset 的 indices_by_bin
        # 只保留在 subset 中的索引，并转换为相对索引
        filtered_indices_by_bin = {}

        for bin_id, indices in base_dataset.indices_by_bin.items():
            # 过滤出在 subset 中的索引，并转换为 Subset 的相对索引
            filtered_indices = []
            for idx in indices:
                if idx in original_to_subset_idx:
                    filtered_indices.append(original_to_subset_idx[idx])

            if len(filtered_indices) > 0:
                filtered_indices_by_bin[bin_id] = np.array(filtered_indices)

        # 创建一个临时的 dataset 对象，只用于存储 indices_by_bin
        class TempDataset:
            def __init__(self, indices_by_bin):
                self.indices_by_bin = indices_by_bin

        temp_dataset = TempDataset(filtered_indices_by_bin)

        # 调用父类初始化（但使用临时 dataset）
        self.dataset = temp_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last


def train_one_epoch(model, train_loader, batch_processor, gradient_bank, optimizer, device, config, epoch, scaler=None):
    """
    训练一个 epoch

    Args:
        model: R-STMRF 模型
        train_loader: 训练数据 loader
        batch_processor: 批次处理器
        gradient_bank: TecGradientBank 实例（预计算梯度库）
        optimizer: 优化器
        device: 设备
        config: 配置字典
        epoch: 当前 epoch（从 0 开始）
        scaler: GradScaler for AMP (自动混合精度)

    Returns:
        avg_loss: 平均损失
        loss_dict: 各项损失的详细字典
            - mse: 纯预测误差（不受不确定性影响）
            - nll: 异方差损失（可能为负）
            - total: 总优化目标
            - chapman, tec_direction: 物理损失分项
            - physics_total: 物理损失总和
    """
    model.train()

    # 不确定性 Warm-up 逻辑
    warmup_epochs = config.get('uncertainty_warmup_epochs', 5)
    use_uncertainty = config.get('use_uncertainty', True) and (epoch >= warmup_epochs)

    if epoch < warmup_epochs and config.get('use_uncertainty', True):
        print(f"  [Warm-up] Epoch {epoch+1}/{warmup_epochs}: 使用 MSE+物理损失")

    # 统计变量
    total_loss = 0.0
    total_mse = 0.0  # 纯 MSE（始终计算）
    total_nll = 0.0  # NLL 损失（可能为负）
    total_physics = 0.0
    total_chapman = 0.0
    total_tec_direction = 0.0
    num_batches = 0

    use_amp = config.get('use_amp', False) and scaler is not None

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)

    for batch_idx, batch_data in enumerate(pbar):
        # 1. 处理批次数据（获取序列）
        coords, target_ne, sw_seq = batch_processor.process_batch(batch_data)

        # 2. 查询预计算的 TEC 梯度方向（快速查询，无 ConvLSTM 计算）
        timestamps = coords[:, 3]  # 提取时间维度 [Batch]
        tec_grad_direction = gradient_bank.get_interpolated_gradient(timestamps)  # [Batch, 2, H, W]

        # 判断是否需要计算物理损失（间歇性计算以加速训练）
        physics_loss_freq = config.get('physics_loss_freq', 10)  # 默认每10个batch计算一次
        compute_physics = (batch_idx % physics_loss_freq == 0)

        # 只在需要物理损失时启用梯度
        if compute_physics:
            coords.requires_grad_(True)

        # 3. 前向传播（支持AMP）
        with torch.amp.autocast('cuda', enabled=use_amp):
            pred_ne, log_var, correction, extras = model(coords, sw_seq, tec_grad_direction)

            # ==================== 3.1 始终计算纯 MSE（用于监控精度）====================
            pure_mse = F.mse_loss(pred_ne, target_ne)

            # ==================== 3.2 计算主损失（根据 warm-up 状态选择）====================
            if use_uncertainty:
                # Warm-up 结束后：使用异方差损失（NLL）
                # 1. 约束 log_var 范围（防止崩塌）
                log_var_clamped = torch.clamp(
                    log_var,
                    min=config.get('log_var_min', -10.0),
                    max=config.get('log_var_max', 10.0)
                )

                # 2. 计算 NLL（可能为负）
                precision = torch.exp(-log_var_clamped)
                mse_term = (pred_ne - target_ne) ** 2
                nll_loss = torch.mean(0.5 * precision * mse_term + 0.5 * log_var_clamped)

                # 3. 添加 log_var 正则化（惩罚极端方差，鼓励接近 1）
                log_var_reg = config.get('log_var_regularization', 0.001)
                log_var_penalty = log_var_reg * (log_var_clamped ** 2).mean()

                loss_main = nll_loss + log_var_penalty
            else:
                # Warm-up 期间或配置关闭：使用纯 MSE
                loss_main = pure_mse
                nll_loss = pure_mse  # 用于记录（此时 NLL = MSE）

        # 4. 计算物理约束损失（间歇性计算）
        # ⚠️ 关键：物理损失包含二阶导数，必须在AMP外计算（已在physics_losses中禁用AMP）
        if compute_physics:
            # 物理损失计算已在函数内部禁用AMP（见chapman_smoothness_loss）
            loss_physics, physics_dict = combined_physics_loss(
                pred_ne=pred_ne,
                coords=coords,
                tec_grad_direction=extras.get('tec_grad_direction'),
                coords_normalized=extras.get('coords_normalized'),
                w_chapman=config['w_chapman'],
                w_tec_direction=config.get('w_tec_direction', 0.05),
                tec_lat_range=config['lat_range'],
                tec_lon_range=config['lon_range']
            )
        else:
            # 跳过物理损失计算，使用零损失
            loss_physics = 0.0
            physics_dict = {
                'physics_total': 0.0,
                'chapman': 0.0,
                'tec_direction': 0.0
            }

        # 5. 总损失
        loss = config['w_mse'] * loss_main + loss_physics

        # 6. 反向传播（支持AMP）
        optimizer.zero_grad()

        if use_amp:
            # AMP反向传播
            scaler.scale(loss).backward()

            # 梯度裁剪
            if config['grad_clip'] is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])

            scaler.step(optimizer)
            scaler.update()
        else:
            # 标准反向传播
            loss.backward()

            # 梯度裁剪
            if config['grad_clip'] is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])

            optimizer.step()

        # 7. 统计
        total_loss += loss.item()
        total_mse += pure_mse.item()  # 纯 MSE
        total_nll += nll_loss.item() if use_uncertainty else pure_mse.item()  # NLL
        total_physics += physics_dict['physics_total']
        total_chapman += physics_dict['chapman']
        total_tec_direction += physics_dict.get('tec_direction', 0.0)
        num_batches += 1

        # 更新进度条
        physics_str = f"{physics_dict['physics_total']:.4f}" if compute_physics else "skip"
        uncertainty_str = "NLL" if use_uncertainty else "MSE"
        pbar.set_postfix({
            'Loss': f"{loss.item():.4f}",
            'Pure_MSE': f"{pure_mse.item():.4f}",
            'Mode': uncertainty_str,
            'Physics': physics_str
        })

    # 平均损失
    avg_loss = total_loss / num_batches
    loss_dict = {
        'total': avg_loss,                          # 总优化目标
        'mse': total_mse / num_batches,             # 纯 MSE（精度监控）
        'nll': total_nll / num_batches,             # NLL 损失（可能为负）
        'physics_total': total_physics / num_batches,  # 物理损失总和
        'chapman': total_chapman / num_batches,        # Chapman 平滑
        'tec_direction': total_tec_direction / num_batches  # TEC 方向
    }

    return avg_loss, loss_dict


def validate(model, val_loader, batch_processor, gradient_bank, device, config):
    """
    验证模型

    Args:
        model: R-STMRF 模型
        val_loader: 验证数据 loader
        batch_processor: 批次处理器
        gradient_bank: TecGradientBank 实例（预计算梯度库）
        device: 设备
        config: 配置字典

    Returns:
        avg_loss: 平均损失
        metrics: 评估指标字典
    """
    model.eval()

    total_loss = 0.0
    total_mse = 0.0
    num_batches = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_data in tqdm(val_loader, desc="Validating", leave=False):
            # 处理批次数据
            coords, target_ne, sw_seq = batch_processor.process_batch(batch_data)

            # 查询预计算的 TEC 梯度方向
            timestamps = coords[:, 3]  # 提取时间维度 [Batch]
            tec_grad_direction = gradient_bank.get_interpolated_gradient(timestamps)  # [Batch, 2, H, W]

            # 前向传播
            pred_ne, log_var, correction, extras = model(coords, sw_seq, tec_grad_direction)

            # MSE 损失
            loss_mse = F.mse_loss(pred_ne, target_ne)

            total_loss += loss_mse.item()
            total_mse += loss_mse.item()
            num_batches += 1

            # 收集预测和真值（用于计算指标）
            all_preds.append(pred_ne.cpu())
            all_targets.append(target_ne.cpu())

    # 计算平均损失
    avg_loss = total_loss / num_batches

    # 计算评估指标
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    mae = np.mean(np.abs(all_preds - all_targets))
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    r2 = 1 - np.sum((all_preds - all_targets) ** 2) / np.sum((all_targets - np.mean(all_targets)) ** 2)

    metrics = {
        'loss': avg_loss,
        'mse': total_mse / num_batches,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }

    return avg_loss, metrics


def train_r_stmrf(config):
    """
    主训练函数

    Args:
        config: 配置字典

    Returns:
        model: 训练好的模型
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        相关管理器
    """
    # 设置随机种子
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    device = torch.device(config['device'])
    print(f"\n{'='*7}")
    print(f"R-STMRF 训练流程")
    print(f"使用设备: {device}")
    print(f"{'='*7}\n")

    # ==================== 1. 初始化数据管理器 ====================
    print("[步骤 1] 初始化数据管理器...")

    sw_manager = SpaceWeatherManager(
        txt_path=config['sw_path'],
        start_date_str=config['start_date_str'],
        total_hours=config['total_hours'],
        seq_len=config['seq_len'],
        device=device
    )

    tec_manager = TECDataManager(
        tec_map_path=config['tec_path'],
        total_hours=config['total_hours'],
        seq_len=config['seq_len'],
        device=device
        # 保持原始分辨率73×73，不再使用降采样
    )

    # ==================== 2. 加载 IRI 神经代理 ====================
    print("\n[步骤 2] 加载 IRI 神经代理场...")

    if not os.path.exists(config['iri_proxy_path']):
        raise FileNotFoundError(f"IRI 代理未找到: {config['iri_proxy_path']}")

    iri_proxy = IRINeuralProxy(layers=[4, 128, 128, 128, 128, 1]).to(device)
    state_dict = torch.load(config['iri_proxy_path'], map_location=device)
    iri_proxy.load_state_dict(state_dict)
    iri_proxy.eval()
    print("  ✓ IRI 代理已加载并冻结")

    # ==================== 3. 准备数据集（随机划分）====================
    print("\n[步骤 3] 准备数据集（随机划分）...")

    # 内存映射设置
    use_memmap = config.get('use_memmap', False)
    if use_memmap:
        print(f"  按需从磁盘读取")
    else:
        print(f"  一次性加载到内存")

    # 加载全部数据
    full_dataset = FY3D_Dataset(
        npy_path=config['fy_path'],
        mode='train',
        val_days=[],  # 不使用日期过滤，加载全部数据
        bin_size_hours=config['bin_size_hours'],
        use_memmap=use_memmap
    )

    # 随机划分
    total_samples = len(full_dataset)
    val_size = int(total_samples * config['val_ratio'])
    train_size = total_samples - val_size

    print(f"  总样本: {total_samples}")
    print(f"  训练集: {train_size} | 验证集: {val_size}")

    generator = torch.Generator().manual_seed(config['seed'])
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    # 通过时间分箱，确保每个 batch 内的样本来自相似的时间窗口
    train_sampler = SubsetTimeBinSampler(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        drop_last=False
    )
    val_sampler = SubsetTimeBinSampler(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        drop_last=False
    )

    # 创建 DataLoader（使用 batch_sampler + 性能优化）
    # 性能优化参数：
    # - pin_memory: 加速CPU-GPU传输
    # - prefetch_factor: 预取batch数量
    # - persistent_workers: 保持worker进程（减少启动开销）
    dataloader_kwargs = {
        'num_workers': config.get('num_workers', 0),
        'pin_memory': config.get('pin_memory', device.type == 'cuda'),
    }

    # 添加prefetch参数（仅当num_workers > 0时）
    if config.get('num_workers', 0) > 0:
        dataloader_kwargs['prefetch_factor'] = config.get('prefetch_factor', 2)
        dataloader_kwargs['persistent_workers'] = config.get('persistent_workers', False)

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        **dataloader_kwargs
    )

    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        **dataloader_kwargs
    )

    print(f"  训练批次: {len(train_loader)} | 验证批次: {len(val_loader)}")

    # 创建批次处理器
    batch_processor = SlidingWindowBatchProcessor(sw_manager, tec_manager, device)

    # ==================== 4. 初始化模型 ====================
    print("\n[步骤 4] 初始化 R-STMRF 模型...")

    model = R_STMRF_Model(
        iri_proxy=iri_proxy,
        lat_range=config['lat_range'],
        lon_range=config['lon_range'],
        alt_range=config['alt_range'],
        sw_manager=sw_manager,
        tec_manager=tec_manager,
        start_date_str=config['start_date_str'],
        config=config
    ).to(device)

    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ==================== 4.5. 初始化 TEC 梯度库（新架构）====================
    print("\n[步骤 4.5] 初始化 TEC 梯度库（离线预计算 + 快速查询）...")

    gradient_bank_path = config.get('gradient_bank_path')
    if gradient_bank_path is None:
        raise ValueError("配置中缺少 'gradient_bank_path' 参数！\n"
                        "请先运行 precompute_tec_gradient_bank.py 生成梯度库，"
                        "然后在配置中指定路径。")

    gradient_bank = TecGradientBank(
        gradient_bank_path=gradient_bank_path,
        total_hours=config['total_hours'],
        device=device
    )

    print(f"  ✓ TEC 梯度库加载成功")

    # ==================== 5. 优化器和调度器 ====================
    print("\n[步骤 5] 配置优化器...")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )

    # 学习率调度器
    if config['scheduler_type'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['epochs'],
            eta_min=config['min_lr']
        )
    else:
        scheduler = None

    # 混合精度训练（AMP）
    use_amp = config.get('use_amp', False)
    scaler = None
    if use_amp:
        if device.type == 'cuda':
            scaler = torch.cuda.amp.GradScaler()
            print(f"  启用混合精度训练（AMP）")
        else:
            print(f"  AMP仅支持CUDA设备，已禁用")
            use_amp = False

    # ==================== 6. 训练循环 ====================
    print("\n[步骤 6] 开始训练...")
    physics_freq = config.get('physics_loss_freq', 10)
    if physics_freq > 1:
        print(f"  物理损失每 {physics_freq} 个batch计算一次")
    else:
        print(f"  物理损失每个batch计算一次")

    # 不确定性 Warm-up 提示
    warmup_epochs = config.get('uncertainty_warmup_epochs', 5)
    if config.get('use_uncertainty', True):
        print(f"  前 {warmup_epochs} 个 epoch 使用纯 MSE")
        print(f"  之后启用异方差损失（NLL），学习预测方差")

    print(f"{'='*7}\n")

    # 训练历史记录
    train_losses = []
    val_losses = []
    history = []  # 详细历史数据（用于三视图绘图和保存）
    best_val_loss = float('inf')
    patience_counter = 0

    # 导入绘图函数
    from .plotting import plot_training_curves_3panel

    for epoch in range(config['epochs']):
        print(f"\n{'='*7}")
        print(f"Epoch {epoch+1}/{config['epochs']}")
        print(f"{'='*7}")

        # 训练
        train_loss, train_dict = train_one_epoch(
            model, train_loader, batch_processor, gradient_bank, optimizer, device, config, epoch, scaler
        )
        train_losses.append(train_loss)

        # 验证
        val_loss, val_metrics = validate(model, val_loader, batch_processor, gradient_bank, device, config)
        val_losses.append(val_loss)

        # 打印结果
        print(f"\nEpoch {epoch+1} 结果:")
        print(f"  训练损失: {train_loss:.6f}")
        print(f"    - Pure MSE: {train_dict['mse']:.6f}")
        print(f"    - NLL: {train_dict['nll']:.6f}")
        print(f"    - Physics: {train_dict['physics_total']:.6f}")
        print(f"      · Chapman: {train_dict['chapman']:.6f}")
        print(f"      · TEC Direction: {train_dict['tec_direction']:.6f}")
        print(f"  验证损失: {val_loss:.6f}")
        print(f"    - MAE: {val_metrics['mae']:.6f}")
        print(f"    - RMSE: {val_metrics['rmse']:.6f}")
        print(f"    - R²: {val_metrics['r2']:.4f}")

        # 收集历史数据
        history_record = {
            'epoch': epoch + 1,
            'train_mse': train_dict['mse'],
            'val_mse': val_loss,  # 验证集使用纯 MSE
            'train_nll': train_dict['nll'],
            'total_loss': train_dict['total'],
            'chapman': train_dict['chapman'],
            'tec_direction': train_dict['tec_direction']
        }
        history.append(history_record)

        # 学习率调度
        if scheduler is not None:
            scheduler.step()
            print(f"  当前学习率: {optimizer.param_groups[0]['lr']:.2e}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_path = os.path.join(config['save_dir'], 'best_r_stmrf_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ 保存最佳模型: {save_path}")
        else:
            patience_counter += 1

        # 绘制训练曲线（每个 epoch 更新）
        # 这样可以实时监控训练状态
        plot_training_curves_3panel(
            history,
            save_path=os.path.join(config['save_dir'], 'training_curves_3panel.png')
        )

        # 早停
        if config['early_stopping'] and patience_counter >= config['patience']:
            print(f"\n早停触发！验证损失连续 {config['patience']} 轮未改善")
            break

        # 定期保存检查点
        if (epoch + 1) % config['save_interval'] == 0:
            save_path = os.path.join(config['save_dir'], f'r_stmrf_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), save_path)

    print(f"\n{'='*70}")
    print("训练完成!")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    print(f"{'='*70}\n")

    # ==================== 7. 保存训练历史 ====================
    import json
    history_path = os.path.join(config['save_dir'], 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✓ 训练历史已保存: {history_path}")

    # 最终绘图
    print(f"✓ 最终训练曲线: {os.path.join(config['save_dir'], 'training_curves_3panel.png')}\n")

    return model, train_losses, val_losses, train_loader, val_loader, sw_manager, tec_manager, gradient_bank, batch_processor


# ======================== 主函数 ========================
if __name__ == '__main__':
    # 获取配置
    config = get_config_r_stmrf()

    # 开始训练
    model, train_losses, val_losses, train_loader, val_loader, sw_manager, tec_manager = train_r_stmrf(config)

    print("\n训练脚本执行完毕！")
