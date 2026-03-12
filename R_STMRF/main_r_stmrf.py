"""
R-STMRF 主程序入口

提供训练、评估和可视化的完整流程
"""

import os
import sys

# 添加模块路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import torch

from inr_modules.r_stmrf.config_r_stmrf import get_config_r_stmrf, print_config_r_stmrf
from inr_modules.r_stmrf.train_r_stmrf import train_r_stmrf


def main():
    """主函数"""
    # 获取配置
    config = get_config_r_stmrf()

    # ==================== 1. 训练模型 ====================
    print("\n" + "="*7)
    print("阶段 1: 模型训练")
    print("="*7 + "\n")

    model, train_losses, val_losses, train_loader, val_loader, sw_manager, tec_manager, gradient_bank, batch_processor = train_r_stmrf(config)

    device = torch.device(config['device'])

    # ==================== 2. 评估模型 ====================
    print("\n" + "="*7)
    print("阶段 2: 模型评估")
    print("="*7 + "\n")

    # 加载最佳模型
    best_model_path = os.path.join(config['save_dir'], 'best_r_stmrf_model.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"✓ 已加载最佳模型: {best_model_path}")
    else:
        print("警告: 未找到最佳模型，使用当前模型")

    # 详细评估
    print("\n[2.1] 计算详细评估指标...")
    from inr_modules.r_stmrf.evaluation_r_stmrf import evaluate_r_stmrf_model
    evaluate_r_stmrf_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        batch_processor=batch_processor,
        gradient_bank=gradient_bank,
        device=device,
        save_dir=config['save_dir']
    )

    # ==================== 3. 可视化结果 ====================
    print("\n" + "="*7)
    print("阶段 3: 可视化")
    print("="*7 + "\n")

    # 绘制损失曲线
    print("[3.1] 绘制损失曲线...")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('R-STMRF Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    loss_curve_path = os.path.join(config['save_dir'], 'loss_curve_r_stmrf.png')
    plt.savefig(loss_curve_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 损失曲线已保存: {loss_curve_path}")

    # Parity 图（散点图和密度图）
    print("\n[3.2] 绘制 Parity 图...")
    from inr_modules.r_stmrf.evaluation_r_stmrf import plot_r_stmrf_parity
    plot_r_stmrf_parity(
        model=model,
        val_loader=val_loader,
        batch_processor=batch_processor,
        gradient_bank=gradient_bank,
        device=device,
        save_dir=config['save_dir'],
        config=config
    )

    # 高度剖面可视化
    print("\n[3.3] 绘制高度剖面...")
    from inr_modules.r_stmrf.evaluation_r_stmrf import plot_r_stmrf_altitude_profile

    # 从配置读取可视化参数
    plot_days = config.get('plot_days', [5, 15, 25])
    plot_hours = config.get('plot_hours', [0.0, 6.0, 12.0, 18.0])

    print(f"  绘制时刻: 天数={plot_days}, 小时={plot_hours}")
    print(f"  共生成 {len(plot_days) * len(plot_hours)} 个高度剖面图")

    # 生成所有 (day, hour) 组合
    for day in plot_days:
        for hour in plot_hours:
            plot_r_stmrf_altitude_profile(
                model=model,
                sw_manager=sw_manager,
                gradient_bank=gradient_bank,
                device=device,
                target_day=day,
                target_hour=hour,
                save_dir=config['save_dir'],
                config=config
            )

    # ==================== 4. 完成 ====================
    print("\n" + "="*7)
    print("所有任务完成!")
    print(f"结果保存在: {os.path.abspath(config['save_dir'])}")
    print("="*7 + "\n")

    print("生成的文件:")
    print("  [模型权重]")
    print("    - best_r_stmrf_model.pth          (最佳模型权重)")
    print("    - r_stmrf_epoch_*.pth             (定期保存的检查点)")
    print("  [评估报告]")
    print("    - r_stmrf_evaluation_report.txt   (详细评估指标)")
    print("  [可视化图表]")
    print("    - loss_curve_r_stmrf.png          (训练/验证损失曲线)")
    print("    - parity_scatter_r_stmrf.png      (Parity 散点图)")
    print("    - parity_density_r_stmrf.png      (Parity 密度图)")
    print("    - altitude_profile_*.png          (高度剖面切片图)")
    print()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断训练")
    except Exception as e:
        print(f"\n\n发生错误: {e}")
        import traceback
        traceback.print_exc()
