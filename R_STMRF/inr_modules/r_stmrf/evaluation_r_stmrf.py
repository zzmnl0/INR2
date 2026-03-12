"""
R-STMRF 评估与可视化工具

适配新架构的评估函数（TecGradientBank + 4个返回值）
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
import os


def evaluate_r_stmrf_model(model, train_loader, val_loader, batch_processor,
                           gradient_bank, device, save_dir):
    """
    计算 R-STMRF 模型的详细评估指标并保存报告

    Args:
        model: R-STMRF 模型
        train_loader: 训练集 DataLoader
        val_loader: 验证集 DataLoader
        batch_processor: SlidingWindowBatchProcessor
        gradient_bank: TecGradientBank 实例
        device: 计算设备
        save_dir: 保存目录
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    report_path = os.path.join(save_dir, 'r_stmrf_evaluation_report.txt')

    def get_metrics(dataloader, name):
        """计算单个数据集的指标"""
        print(f"  评估 {name} 集...")
        obs_list, pred_list, iri_list = [], [], []

        with torch.no_grad():
            # 添加 tqdm 进度条，避免看起来像卡住
            from tqdm import tqdm
            for batch_data in tqdm(dataloader, desc=f"    处理 {name} 集", leave=False):
                # 使用 batch_processor 处理数据
                coords, target_ne, sw_seq = batch_processor.process_batch(batch_data)

                # 获取 TEC 梯度方向（从 gradient_bank）
                timestamps = coords[:, 3]  # Time 列
                tec_grad_direction = gradient_bank.get_interpolated_gradient(timestamps)

                # 获取 IRI 背景
                iri_bg = model.get_background(
                    coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
                ).squeeze()

                # 模型预测（新 API: 返回 4 个值）
                ne_pred, log_var, correction, extras = model(coords, sw_seq, tec_grad_direction)
                ne_pred = ne_pred.squeeze()

                obs_list.append(target_ne.squeeze().cpu().numpy())
                pred_list.append(ne_pred.cpu().numpy())
                iri_list.append(iri_bg.cpu().numpy())

        obs = np.concatenate(obs_list)
        pred = np.concatenate(pred_list)
        iri = np.concatenate(iri_list)

        # 计算指标
        rmse = np.sqrt(mean_squared_error(obs, pred))
        r2 = r2_score(obs, pred)
        r_corr, _ = pearsonr(obs, pred)

        # IRI 基线指标
        rmse_iri = np.sqrt(mean_squared_error(obs, iri))
        r2_iri = r2_score(obs, iri)
        r_corr_iri, _ = pearsonr(obs, iri)

        return {
            'rmse': rmse, 'r2': r2, 'r': r_corr,
            'rmse_iri': rmse_iri, 'r2_iri': r2_iri, 'r_iri': r_corr_iri,
            'n_samples': len(obs)
        }

    # 计算指标
    train_metrics = get_metrics(train_loader, "训练")
    val_metrics = get_metrics(val_loader, "验证")

    # 生成报告
    content = []
    content.append("=" * 70)
    content.append("   R-STMRF 电离层重构评估报告")
    content.append("=" * 70)
    content.append(f"模型架构: R-STMRF v2.0+ (TEC Gradient Bank + SIREN)")
    content.append(f"物理约束: Chapman 平滑 + TEC 梯度方向一致性")
    content.append(f"总样本数: {train_metrics['n_samples'] + val_metrics['n_samples']}")
    content.append(f"  训练集: {train_metrics['n_samples']}")
    content.append(f"  验证集: {val_metrics['n_samples']}")
    content.append("-" * 70)
    content.append(f"[训练集指标]")
    content.append(f"  R-STMRF:")
    content.append(f"    RMSE : {train_metrics['rmse']:.5f}")
    content.append(f"    R²   : {train_metrics['r2']:.5f}")
    content.append(f"    R    : {train_metrics['r']:.5f}")
    content.append(f"  IRI-2020 基线:")
    content.append(f"    RMSE : {train_metrics['rmse_iri']:.5f}")
    content.append(f"    R²   : {train_metrics['r2_iri']:.5f}")
    content.append(f"    R    : {train_metrics['r_iri']:.5f}")
    content.append(f"  改进率: {(1 - train_metrics['rmse'] / train_metrics['rmse_iri']) * 100:.2f}%")
    content.append("-" * 70)
    content.append(f"[验证集指标]")
    content.append(f"  R-STMRF:")
    content.append(f"    RMSE : {val_metrics['rmse']:.5f}")
    content.append(f"    R²   : {val_metrics['r2']:.5f}")
    content.append(f"    R    : {val_metrics['r']:.5f}")
    content.append(f"  IRI-2020 基线:")
    content.append(f"    RMSE : {val_metrics['rmse_iri']:.5f}")
    content.append(f"    R²   : {val_metrics['r2_iri']:.5f}")
    content.append(f"    R    : {val_metrics['r_iri']:.5f}")
    content.append(f"  改进率: {(1 - val_metrics['rmse'] / val_metrics['rmse_iri']) * 100:.2f}%")
    content.append("=" * 70)

    report_text = "\n".join(content)
    print(report_text)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\n✓ 评估报告已保存: {report_path}")


def plot_r_stmrf_parity(model, val_loader, batch_processor, gradient_bank,
                        device, save_dir, config):
    """
    绘制 R-STMRF 的 Parity 图（散点图和密度图）

    Args:
        model: R-STMRF 模型
        val_loader: 验证集 DataLoader
        batch_processor: SlidingWindowBatchProcessor
        gradient_bank: TecGradientBank 实例
        device: 计算设备
        save_dir: 保存目录
        config: 配置字典
    """
    model.eval()
    obs, iri, inr = [], [], []

    print("  生成 Parity 图数据...")

    with torch.no_grad():
        from tqdm import tqdm
        for batch_data in tqdm(val_loader, desc="    采样数据", leave=False):
            # 处理批次数据
            coords, target_ne, sw_seq = batch_processor.process_batch(batch_data)

            # 获取 TEC 梯度方向
            timestamps = coords[:, 3]
            tec_grad_direction = gradient_bank.get_interpolated_gradient(timestamps)

            # IRI 背景
            iri_bg = model.get_background(
                coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
            ).squeeze()

            # R-STMRF 预测
            ne_pred, log_var, correction, extras = model(coords, sw_seq, tec_grad_direction)
            ne_pred = ne_pred.squeeze()

            obs.append(target_ne.squeeze().cpu().numpy())
            iri.append(iri_bg.cpu().numpy())
            inr.append(ne_pred.cpu().numpy())

    obs = np.concatenate(obs)
    iri = np.concatenate(iri)
    inr = np.concatenate(inr)

    print(f"  处理点数: {len(obs):,}")

    # 计算指标
    def calc_metrics(y_true, y_pred):
        rmse = np.sqrt(((y_true - y_pred)**2).mean())
        r2 = r2_score(y_true, y_pred)
        r_corr, _ = pearsonr(y_true, y_pred)
        return rmse, r2, r_corr

    rmse_iri, r2_iri, r_iri = calc_metrics(obs, iri)
    rmse_inr, r2_inr, r_inr = calc_metrics(obs, inr)

    # 坐标轴范围
    global_min = min(obs.min(), iri.min(), inr.min())
    global_max = max(obs.max(), iri.max(), inr.max())
    axis_min = np.floor(global_min * 10) / 10
    axis_max = np.ceil(global_max * 10) / 10

    title_iri = (f'IRI-2020 vs Observations\n'
                 f'RMSE={rmse_iri:.4f} | R²={r2_iri:.4f} | R={r_iri:.4f}')
    title_inr = (f'R-STMRF (TEC Gradient Bank) vs Observations\n'
                 f'RMSE={rmse_inr:.4f} | R²={r2_inr:.4f} | R={r_inr:.4f}')

    # Parity plot
    fig1, ax1 = plt.subplots(1, 2, figsize=(16, 7), dpi=150)

    def draw_scatter(axis, x, y, title_text, color):
        axis.scatter(x, y, alpha=0.05, s=0.5, c=color, rasterized=True)
        axis.plot([axis_min, axis_max], [axis_min, axis_max],
                 'r--', lw=2, alpha=0.8, label='1:1 Line')
        axis.set_title(title_text, fontsize=12, fontweight='bold')
        axis.set_xlabel(r'Observed $N_e$ ($\log_{10}$)', fontsize=11)
        axis.set_ylabel(r'Model $N_e$ ($\log_{10}$)', fontsize=11)
        axis.set_xlim(axis_min, axis_max)
        axis.set_ylim(axis_min, axis_max)
        axis.set_aspect('equal')
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=10)

    draw_scatter(ax1[0], obs, iri, title_iri, 'blue')
    draw_scatter(ax1[1], obs, inr, title_inr, 'green')

    plt.tight_layout()
    scatter_path = os.path.join(save_dir, 'parity_scatter_r_stmrf.png')
    plt.savefig(scatter_path, dpi=150)
    plt.close(fig1)
    print(f"  ✓ Scatter plot saved: {scatter_path}")

    # Density plot
    fig2, ax2 = plt.subplots(1, 2, figsize=(17, 7), dpi=150)
    plot_range = [[axis_min, axis_max], [axis_min, axis_max]]
    bins = 400

    def draw_density(axis, x, y, title_text):
        h = axis.hist2d(x, y, bins=bins, range=plot_range,
                       cmap='turbo', norm=LogNorm(), cmin=1)
        axis.plot([axis_min, axis_max], [axis_min, axis_max],
                 'r--', lw=1.5, alpha=0.8, label='1:1 Line')
        axis.set_title(title_text, fontsize=12, fontweight='bold')
        axis.set_xlabel(r'Observed $N_e$ ($\log_{10}$)', fontsize=11)
        axis.set_ylabel(r'Model $N_e$ ($\log_{10}$)', fontsize=11)
        axis.set_aspect('equal')
        axis.grid(True, linestyle=':', alpha=0.4)
        axis.legend(fontsize=10)

        divider = make_axes_locatable(axis)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        plt.colorbar(h[3], cax=cax, label='Counts (Log Scale)')

    draw_density(ax2[0], obs, iri, title_iri)
    draw_density(ax2[1], obs, inr, title_inr)

    plt.tight_layout()
    density_path = os.path.join(save_dir, 'parity_density_r_stmrf.png')
    plt.savefig(density_path, dpi=150)
    plt.close(fig2)
    print(f"  ✓ Density plot saved: {density_path}")


def plot_r_stmrf_altitude_profile(model, sw_manager, gradient_bank, device,
                                   target_day, target_hour, save_dir, config):
    """
    绘制 R-STMRF 的全高度层切片图

    Args:
        model: R-STMRF 模型
        sw_manager: SpaceWeatherManager
        gradient_bank: TecGradientBank 实例
        device: 计算设备
        target_day: 目标日期
        target_hour: 目标小时
        save_dir: 保存目录
        config: 配置字典
    """
    model.eval()
    global_time = target_day * 24.0 + target_hour

    # 获取 SW 序列
    time_tensor = torch.tensor([global_time], device=device)
    sw_seq = sw_manager.get_drivers_sequence(time_tensor)

    # 提取驱动值用于标题
    kp_curr = sw_seq[0, -1, 0].item()
    f107_curr = sw_seq[0, -1, 1].item()

    kp_disp = (kp_curr + 1.0) / 2.0 * 9.0
    f107_disp = f107_curr * 60.0 + 210.0

    print(f"  绘制剖面 Day {target_day} Hour {target_hour:02.0f} "
          f"(Kp={kp_disp:.1f}, F10.7={f107_disp:.1f})")

    # 定义高度层
    alt_levels = list(range(120, 500, 30))
    num_alts = len(alt_levels)

    # 空间网格
    lat_grid = np.linspace(-90, 90, 91)
    lon_grid = np.linspace(-180, 180, 180)
    LON, LAT = np.meshgrid(lon_grid, lat_grid)
    extent = [-180, 180, -90, 90]

    # 创建图形
    fig, axes = plt.subplots(nrows=num_alts, ncols=3,
                            figsize=(18, 2.5 * num_alts))

    for i, alt in enumerate(alt_levels):
        ax_row = axes[i] if num_alts > 1 else [axes[0], axes[1], axes[2]]

        # 构建查询坐标
        lat_flat = LAT.flatten()
        lon_flat = LON.flatten()
        alt_flat = np.full_like(lat_flat, alt)
        time_flat = np.full_like(lat_flat, global_time)

        coords = np.stack([lat_flat, lon_flat, alt_flat, time_flat], axis=1).astype(np.float32)
        coords_t = torch.from_numpy(coords).to(device)
        batch_len = coords_t.shape[0]

        # 构建输入
        sw_t = sw_seq.expand(batch_len, -1, -1)
        tec_grad_t = gradient_bank.get_interpolated_gradient(coords_t[:, 3])

        # 推理
        with torch.no_grad():
            iri_bg = model.get_background(
                coords_t[:, 0], coords_t[:, 1], coords_t[:, 2], coords_t[:, 3]
            ).squeeze().cpu().numpy()

            ne_pred, log_var, correction, extras = model(coords_t, sw_t, tec_grad_t)
            inr_pred = ne_pred.squeeze().cpu().numpy()

            iri_map = iri_bg.reshape(LAT.shape)
            inr_map = inr_pred.reshape(LAT.shape)
            res_map = inr_map - iri_map

        # 颜色范围
        local_min = max(min(np.nanmin(iri_map), np.nanmin(inr_map)), 8.0)
        local_max = min(max(np.nanmax(iri_map), np.nanmax(inr_map)), 13.0)

        if local_max - local_min < 0.2:
            mid = (local_max + local_min) / 2
            local_max = mid + 0.1
            local_min = mid - 0.1

        local_res_max = max(np.nanmax(np.abs(res_map)), 0.2)

        # 绘制三列
        im1 = ax_row[0].imshow(iri_map, extent=extent, origin='lower',
                               cmap='plasma', vmin=local_min, vmax=local_max,
                               aspect='auto')
        ax_row[0].set_ylabel(f'{alt} km\nLat', fontweight='bold')
        if i == 0:
            ax_row[0].set_title('IRI-2020 Background', fontsize=12)
        plt.colorbar(im1, ax=ax_row[0], pad=0.02)

        im2 = ax_row[1].imshow(inr_map, extent=extent, origin='lower',
                               cmap='plasma', vmin=local_min, vmax=local_max,
                               aspect='auto')
        if i == 0:
            ax_row[1].set_title('R-STMRF Prediction', fontsize=12)
        plt.colorbar(im2, ax=ax_row[1], pad=0.02)

        im3 = ax_row[2].imshow(res_map, extent=extent, origin='lower',
                               cmap='bwr', vmin=-local_res_max, vmax=local_res_max,
                               aspect='auto')
        if i == 0:
            ax_row[2].set_title('Residual', fontsize=12)
        plt.colorbar(im3, ax=ax_row[2], pad=0.02)

        # X轴标签
        if i < num_alts - 1:
            for ax in ax_row:
                ax.set_xticks([])
        else:
            for ax in ax_row:
                ax.set_xlabel('Longitude', fontsize=10)

    # Main title
    plt.suptitle(
        f'R-STMRF Reconstruction Day {target_day}, {target_hour:02.0f}:00 UT '
        f'(Kp={kp_disp:.1f}, F10.7={f107_disp:.1f})\n'
        f'Architecture: TEC Gradient Direction Constraint + Space Weather Modulation',
        fontsize=14, fontweight='bold', y=1.01
    )

    plt.tight_layout()

    filename = f'altitude_profile_day{target_day:02d}_hour{int(target_hour):02d}_r_stmrf.png'
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Altitude profile saved: {save_path}")
