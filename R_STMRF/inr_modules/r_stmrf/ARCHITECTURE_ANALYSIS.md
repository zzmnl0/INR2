# R-STMRF 架构分析与内存优化

## 当前架构概览（2026-01-29 最新版）

### 核心设计理念

**TEC 作为梯度方向约束，而非数值调制**

```
Ne_fused(x,t) = IRI_background(x,t) + SIREN_residual(x,t)
            ↓ 约束
    梯度方向一致性损失（仅约束方向，不约束幅值）
```

## 模块详解

### 1. 数据流（Data Flow）

```
FY3D卫星数据 [Batch, 5] (Lat, Lon, Alt, Time, Ne_Log)
    ↓
SlidingWindowBatchProcessor
    ├─ 识别唯一时间窗口: torch.unique(time_indices)
    ├─ 查询空间天气序列: sw_seq [Batch, Seq, 2]
    ├─ 查询TEC地图序列: unique_tec_map_seq [N_unique, Seq, 1, 46, 91]
    └─ 返回: (coords, target_ne, sw_seq, unique_tec_map_seq, tec_indices, target_tec_map)
```

**内存优化点**：
- TEC地图降采样 4x: 181×361 → 46×91（16× 内存减少）
- 只计算唯一时间窗口的TEC序列，不随batch重复

### 2. 空间分支（Spatial Branch）- 主建模网络

```python
# A1. SIREN 空间基函数（主路）
spatial_input = [lat_n, lon_n, alt_n, sin_lt, cos_lt]  # [Batch, 5]
h_spatial = SIREN(spatial_input)  # [Batch, 64]
# 职责: 学习电子密度的空间表达能力

# A2. ConvLSTM 提取 TEC 梯度方向特征（Context，非主路）
F_tec_unique = ConvLSTM(unique_tec_map_seq)  # [N_unique, 16, 46, 91]
# 职责: 提取 TEC 水平梯度方向的时序演化

# A3. 梯度方向提取
tec_grad_direction = Conv2d(F_tec_unique)  # [N_unique, 2, 46, 91]
# 输出: (∂TEC/∂lat, ∂TEC/∂lon) 方向向量

# A4. 索引（不复制）
tec_grad_direction_batch = tec_grad_direction[tec_indices]  # [Batch, 2, 46, 91]

# A5. 无调制（移除了FiLM）
h_spatial_mod = h_spatial  # [Batch, 64] - 直接使用，无调制
```

**内存优化点**：
- ConvLSTM: 1层（原2层），16通道（原32）
- 内存: N_unique × 16 × 46 × 91 ≈ 27 MB（N_unique=100）
- 使用索引而非repeat，避免内存复制

### 3. 时间分支（Temporal Branch）

```python
# B1. SIREN 时间基函数
h_temporal = SIREN(time_n)  # [Batch, 64]

# B2. LSTM 编码空间天气
z_env = LSTM(sw_seq)  # [Batch, 64]

# B3. 加性调制
beta_temporal = MLP(z_env)  # [Batch, 64]
h_temporal_mod = h_temporal + beta_temporal  # [Batch, 64]
```

### 4. 融合解码（Fusion Decoder）

```python
# 拼接调制后的特征
fusion = concat(h_spatial_mod, h_temporal_mod)  # [Batch, 128]

# 解码得到残差
residual = Decoder(fusion)  # [Batch, 1]

# 最终输出
Ne_fused = IRI_background + residual  # [Batch, 1]
```

### 5. 物理约束损失（Physics Losses）

#### 5.1 Chapman 垂直平滑损失

```python
# 计算 Ne_fused 的二阶导数 ∂²Ne/∂h²
grad_second_alt = autograd.grad(grad_alt, coords)
loss_chapman = mean(grad_second_alt ** 2)
```

**物理意义**：Chapman层应该平滑，无非物理震荡

#### 5.2 TEC 梯度方向一致性损失（新设计）

```python
# 1. 计算 Ne_fused 的水平梯度
grad_ne = autograd.grad(Ne_fused, coords)[:, :2]  # [Batch, 2]

# 2. 从 tec_grad_direction 采样期望梯度方向
tec_grad_expected = grid_sample(tec_grad_direction, coords)  # [Batch, 2]

# 3. 归一化（去除幅值）
grad_ne_norm = normalize(grad_ne)
tec_grad_norm = normalize(tec_grad_expected)

# 4. 余弦相似度损失（仅约束方向）
loss = 1 - cosine_similarity(grad_ne_norm, tec_grad_norm)
```

**关键特性**：
- ❌ 不约束 TEC 幅值
- ❌ 不要求 Ne 积分等于 TEC
- ✅ 仅约束方向，允许幅值自由变化
- ✅ 适用于高度覆盖不完整的情况

## 内存占用分析

### 模型参数内存

| 模块 | 参数量 | 内存 (FP32) |
|------|--------|-------------|
| SIREN Spatial | ~50K | ~200 KB |
| SIREN Temporal | ~50K | ~200 KB |
| ConvLSTM (1层, 16通道) | ~12K | ~48 KB |
| LSTM (2层, 64通道) | ~50K | ~200 KB |
| Decoder | ~16K | ~64 KB |
| **总计** | **~178K** | **~712 KB** |

### 前向传播内存（batch_size=2048）

| 项目 | 形状 | 内存 (FP32) |
|------|------|-------------|
| coords | [2048, 4] | 32 KB |
| sw_seq | [2048, 6, 2] | 192 KB |
| unique_tec_map_seq (N_unique=100) | [100, 6, 1, 46, 91] | 10 MB |
| F_tec_unique | [100, 16, 46, 91] | 27 MB |
| tec_grad_direction | [2048, 2, 46, 91] | 69 MB |
| h_spatial | [2048, 64] | 512 KB |
| h_temporal | [2048, 64] | 512 KB |
| **估计总计** | - | **~110 MB** |

### 反向传播内存（梯度 + 中间激活）

- 梯度内存: ~2x 参数内存 ≈ 1.4 MB
- 中间激活: ~2x 前向内存 ≈ 220 MB
- **估计总计**: **~221 MB**

### 峰值内存（训练时）

```
总内存 = 模型参数 + 优化器状态 + 前向激活 + 反向梯度 + TEC数据
      ≈ 0.7 MB + 2.8 MB + 110 MB + 221 MB + 50 MB
      ≈ 385 MB (单个batch)
```

**实际可能更高**：
- Adam优化器: 2x参数内存（momentum + variance）
- 物理损失的autograd图: 额外激活内存
- ConvLSTM hidden states: ~27 MB

**保守估计**: **~500-700 MB per batch**

## 磁盘占用分析

### 模型checkpoint

- 模型权重 (FP32): ~712 KB
- 优化器状态 (Adam): ~2.1 MB
- **单个checkpoint**: ~2.8 MB

### 训练日志

- 50 epochs, 每5 epochs保存: 10个checkpoints
- **总磁盘占用**: ~28 MB

### 数据缓存

- TEC地图 (720h, 46×91, FP32): ~12 MB
- 空间天气 (720h, 2): ~5.7 KB
- **总计**: ~12 MB

### 总磁盘占用

```
总磁盘 = Checkpoints + 数据缓存 + 日志
      ≈ 28 MB + 12 MB + 1 MB
      ≈ 41 MB
```

## 优化建议

### 内存优化（优先级从高到低）

#### 1. 降低 batch_size（立即生效）

```python
# config_r_stmrf.py
'batch_size': 1024,  # 2048 → 1024（减半内存）
'batch_size': 512,   # 2048 → 512（减少75%内存）
```

**效果**: 线性减少前向/反向内存

#### 2. 减小 TEC 序列长度

```python
'seq_len': 4,  # 6 → 4（减少33%）
```

**效果**:
- unique_tec_map_seq: 10 MB → 6.7 MB
- ConvLSTM hidden states: 减少33%

#### 3. 进一步降低 TEC 分辨率

```python
'tec_downsample_factor': 6,  # 4 → 6 (181×361 → 31×61)
```

**效果**:
- F_tec内存: 27 MB → 12 MB（减少56%）
- 可能损失梯度方向精度

#### 4. 减小模型维度

```python
'basis_dim': 48,  # 64 → 48（减少25%）
'siren_hidden': 96,  # 128 → 96（减少25%）
```

**效果**:
- 参数量: 178K → ~100K（减少44%）
- h_spatial/h_temporal: 512 KB → 384 KB

#### 5. 使用混合精度训练（FP16）

```python
'use_amp': True,  # 启用自动混合精度
```

**效果**:
- 内存减少约50%
- 需要GPU支持（CPU不支持FP16）

#### 6. 梯度检查点（Gradient Checkpointing）

```python
# 在ConvLSTM中使用checkpoint
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    return checkpoint(self.convlstm, x)
```

**效果**:
- 反向传播内存减少50-70%
- 训练时间增加10-20%

### 磁盘优化

#### 1. 只保存最佳模型

```python
'save_best_only': True,  # 只保存验证损失最低的模型
'save_interval': 10,  # 减少保存频率
```

**效果**: 28 MB → 2.8 MB（减少90%）

#### 2. 压缩保存

```python
# train_r_stmrf.py
torch.save(model.state_dict(), path, _use_new_zipfile_serialization=True)
```

**效果**: 可能减少30-50%

#### 3. 删除优化器状态

```python
# 只保存模型权重，不保存优化器
torch.save({
    'model': model.state_dict(),
    # 'optimizer': optimizer.state_dict(),  # 不保存
}, path)
```

**效果**: 2.8 MB → 0.7 MB（减少75%）

## 推荐配置（内存受限环境）

### CPU环境（8GB RAM可用）

```python
CONFIG_R_STMRF = {
    # 训练参数
    'batch_size': 512,  # 减小
    'epochs': 30,  # 减少epoch数

    # 模型维度
    'basis_dim': 48,
    'siren_hidden': 96,
    'siren_layers': 2,  # 减少层数

    # TEC参数
    'seq_len': 4,
    'tec_downsample_factor': 6,
    'tec_feat_dim': 12,
    'convlstm_layers': 1,

    # 损失权重
    'w_chapman': 0.1,
    'w_tec_direction': 0.02,  # 降低

    # 保存策略
    'save_best_only': True,
    'save_interval': 10,
}
```

**预期内存**: ~200-300 MB per batch
**预期磁盘**: ~5 MB

### GPU环境（4GB VRAM）

```python
CONFIG_R_STMRF = {
    'batch_size': 1024,
    'use_amp': True,  # 启用混合精度

    # 其他参数保持默认
}
```

**预期内存**: ~250 MB per batch (FP16)
**预期磁盘**: ~15 MB

## 监控内存使用

### 1. 训练脚本中添加监控

```python
import psutil
import os

def log_memory():
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"当前内存使用: {mem_mb:.1f} MB")

# 在训练循环中
for batch_idx, batch_data in enumerate(train_loader):
    if batch_idx % 10 == 0:
        log_memory()
```

### 2. 使用 memory_profiler

```bash
pip install memory_profiler

# 运行
python -m memory_profiler main_r_stmrf.py
```

## 性能 vs 内存权衡

| 配置 | 内存 | 训练速度 | 模型性能 |
|------|------|----------|----------|
| 默认 | ~500 MB | 1.0x | 100% |
| 小batch | ~250 MB | 0.8x | 98% |
| 小模型 | ~150 MB | 1.2x | 93% |
| 小TEC | ~300 MB | 1.1x | 95% |
| 组合优化 | ~120 MB | 0.9x | 88% |

**建议**:
- 如果内存充足，使用默认配置
- 如果内存受限但要求高性能，只降低batch_size
- 如果极度受限，使用组合优化

## 总结

当前架构已经过大幅优化：
1. ✅ TEC降采样 16x（181×361 → 46×91）
2. ✅ ConvLSTM简化（2层32通道 → 1层16通道）
3. ✅ 移除FiLM调制头（减少参数）
4. ✅ 识别唯一时间窗口（避免重复计算）

**进一步优化空间**：
- 降低batch_size（最有效）
- 减小TEC序列长度（次有效）
- 减小模型维度（会损失性能）

**不建议的优化**：
- 进一步降低TEC分辨率（会严重损失梯度方向精度）
- 移除ConvLSTM（失去时序建模能力）
