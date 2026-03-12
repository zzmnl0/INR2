"""
R-STMRF 训练曲线可视化工具

三视图策略 (3-Panel Layout)：
  1. Top: 精度监控 (Pure MSE) - Log Scale
  2. Middle: 优化目标 (Total Loss & NLL) - Linear Scale
  3. Bottom: 物理约束 (Chapman & TEC) - Log Scale
"""

import matplotlib.pyplot as plt
import numpy as np
import os


def plot_training_curves_3panel(history, save_path='training_curves_3panel.png'):
    """
    绘制 R-STMRF 训练曲线（三视图）

    针对包含 MSE（精度）、NLL（不确定性，可能为负）和物理损失（约束）的复杂损失函数，
    采用三视图分离展示，避免量级和符号差异导致的混乱。

    Args:
        history: 列表，每个元素是每个 epoch 的 loss_dict
                 结构示例: {
                     'epoch': 1,
                     'train_mse': 0.1,      # 纯 MSE（精度）
                     'val_mse': 0.12,       # 验证集 MSE
                     'train_nll': -0.5,     # NLL 损失（可能为负）
                     'total_loss': 0.3,     # 总优化目标
                     'chapman': 0.05,       # Chapman 平滑
                     'tec_direction': 0.01  # TEC 方向约束
                 }
        save_path: 保存路径
    """
    if len(history) == 0:
        print("⚠️  Warning: history is empty, skipping plot")
        return

    epochs = [h['epoch'] for h in history]

    # 提取数据（使用 get 防止 key 不存在报错）
    train_mse = [h.get('train_mse', np.nan) for h in history]
    val_mse = [h.get('val_mse', np.nan) for h in history]

    total_loss = [h.get('total_loss', np.nan) for h in history]
    train_nll = [h.get('train_nll', np.nan) for h in history]

    chapman = [h.get('chapman', np.nan) for h in history]
    tec_direction = [h.get('tec_direction', np.nan) for h in history]

    # 创建画布（3 行 1 列，共享 X 轴）
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

    # ==================== Panel 1: Accuracy Monitoring (Pure MSE) ====================
    # Log scale used for better visualization of MSE convergence
    ax1.plot(epochs, train_mse, label='Train Pure MSE', color='blue', linewidth=2, marker='o', markersize=3)
    ax1.plot(epochs, val_mse, label='Val MSE', color='orange', linestyle='--', linewidth=2, marker='s', markersize=3)
    ax1.set_yscale('log')
    ax1.set_ylabel('MSE (Log Scale)', fontsize=12)
    ax1.set_title('1. Model Accuracy (Pure MSE)\nMonitoring prediction quality without uncertainty interference',
                  fontsize=13, fontweight='bold')
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.legend(loc='upper right', fontsize=11)

    # ==================== Panel 2: Optimization Objective (Total Loss & NLL) ====================
    # Linear scale used because NLL can be negative
    ax2.plot(epochs, total_loss, label='Total Loss (Optimizer Target)', color='black', linewidth=2, marker='o', markersize=3)
    ax2.plot(epochs, train_nll, label='NLL Term (Uncertainty)', color='green', linestyle=':', linewidth=2, alpha=0.7)
    ax2.axhline(y=0, color='red', linestyle='-', linewidth=1, alpha=0.3, label='Zero Line')
    ax2.set_ylabel('Loss Value (Linear)', fontsize=12)
    ax2.set_title('2. Optimization Objective (May Contain Negative Values)\nMonitoring optimizer behavior',
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=11)

    # ==================== Panel 3: Physics Constraints ====================
    # Log scale used for physics losses (fast convergence with varying scales)
    # Filter out zero values (cannot be displayed on log scale)
    chapman_nonzero = [c if c > 0 else np.nan for c in chapman]
    tec_nonzero = [t if t > 0 else np.nan for t in tec_direction]

    ax3.plot(epochs, chapman_nonzero, label='Chapman Smoothness', color='purple', linewidth=2, marker='o', markersize=3)
    ax3.plot(epochs, tec_nonzero, label='TEC Direction Constraint', color='brown', linewidth=2, marker='s', markersize=3)
    ax3.set_yscale('log')
    ax3.set_ylabel('Constraint Loss (Log Scale)', fontsize=12)
    ax3.set_title('3. Physics Constraints\nMonitoring constraint effectiveness and convergence',
                  fontsize=13, fontweight='bold')
    ax3.set_xlabel('Epochs', fontsize=12)
    ax3.grid(True, which="both", ls="-", alpha=0.2)
    ax3.legend(loc='upper right', fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 3-panel training curves saved: {save_path}")


def plot_simple_loss_curve(train_losses, val_losses, save_path='loss_curve_simple.png'):
    """
    绘制简单的训练/验证损失曲线（兼容旧接口）

    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', linewidth=2, color='blue')
    plt.plot(val_losses, label='Val Loss', linewidth=2, color='orange', linestyle='--')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training & Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Simple loss curve saved: {save_path}")


# ======================== 测试代码 ========================
if __name__ == '__main__':
    print("="*60)
    print("R-STMRF 训练曲线绘图测试")
    print("="*60)

    # 模拟训练历史数据
    epochs = 20
    history = []

    for i in range(epochs):
        # 模拟损失下降
        train_mse = 0.5 * np.exp(-i * 0.1) + 0.01
        val_mse = 0.55 * np.exp(-i * 0.1) + 0.015
        train_nll = -0.2 + 0.3 * np.exp(-i * 0.15)  # NLL 可能为负
        total_loss = train_mse + 0.1 * np.exp(-i * 0.2)
        chapman = 0.08 * np.exp(-i * 0.2)
        tec = 0.03 * np.exp(-i * 0.15)

        history.append({
            'epoch': i + 1,
            'train_mse': train_mse,
            'val_mse': val_mse,
            'train_nll': train_nll,
            'total_loss': total_loss,
            'chapman': chapman,
            'tec_direction': tec
        })

    # 测试绘图
    plot_training_curves_3panel(history, save_path='/tmp/test_3panel.png')

    print("\n" + "="*60)
    print("测试完成! 查看 /tmp/test_3panel.png")
    print("="*60)
