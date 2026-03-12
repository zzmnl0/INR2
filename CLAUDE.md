# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **physics-guided implicit neural representation (INR)** system for reconstructing 3D ionospheric electron density (Ne). It fuses multiple data sources — FY satellite observations, IRI model background, TEC maps, and space weather indices (Kp/F10.7) — to improve accuracy over the IRI model baseline.

There are two model variants, each in its own **independent top-level directory** with no shared code:

| 目录 | 模型 | 说明 |
|---|---|---|
| `INR1/` | PhysicsGuidedINR | Fourier特征编码 + Transformer调制 |
| `R_STMRF/` | R-STMRF | SIREN层 + LSTM，TEC仅作梯度方向约束 |

Each directory contains its own `inr_modules/data_managers/` copy — changes to one do **not** affect the other.

## Running the Models

```bash
# Run original PhysicsGuidedINR (train + eval + visualize)
cd INR1
python main_inr.py

# Run R-STMRF model (recommended, more memory-efficient)
cd R_STMRF
python main_r_stmrf.py

# ISR comparison visualization (R-STMRF)
cd R_STMRF
python R_STMRF_ISR_IRI_plot.py
```

### Pre-computing TEC gradient bank (required for R-STMRF, run once)

```bash
cd R_STMRF
python inr_modules/r_stmrf/precompute_tec_gradient_bank.py
```

This generates `tec_gradient_bank.npy` (shape: `(720, 2, 73, 73)`, float16) used in place of online ConvLSTM computation.

## Configuration

**Original INR**: Edit `INR1/inr_modules/config.py` — global `CONFIG` dict, modified via `update_config(**kwargs)`.

**R-STMRF**: Edit `R_STMRF/inr_modules/r_stmrf/config_r_stmrf.py` — global `CONFIG_R_STMRF` dict, modified via `update_config_r_stmrf(**kwargs)`.

**Critical data paths** (must exist before running):
- `fy_path`: FY satellite EDP data `.npy`
- `iri_proxy_path`: Pre-trained IRI neural proxy `.pth`
- `sw_path`: OMNI Kp/F10.7 text file
- `tec_path`: IGS TEC map `.npy` — shape `(720, 71, 73)`
- `gradient_bank_path` (R-STMRF only): Pre-computed TEC gradient bank `.npy`

GPU/CPU is auto-detected at config import time — R-STMRF automatically sets `batch_size`, `num_workers`, AMP, and `pin_memory`.

## Architecture

### Data Managers (duplicated independently in each project)

Both `INR1/inr_modules/data_managers/` and `R_STMRF/inr_modules/data_managers/` contain the same 4 files:
- **IRI Neural Proxy** (`irinc_neural_proxy.py`): Frozen neural network approximating IRI model output. Parameters are never updated during training.
- **SpaceWeatherManager** (`space_weather_manager.py`): Loads Kp/F10.7, generates sliding-window sequences `[Batch, SeqLen, 2]`.
- **TECDataManager** (`tec_manager.py`): Loads TEC maps, supports spatiotemporal interpolation via `grid_sample`, exposes `get_tec_sequence()` and `get_tec_map_sequence()`.
- **FY Dataloader** (`FY_dataloader.py`): Loads FY satellite electron density profile observations; `TimeBinSampler` groups batches by time window.

### PhysicsGuidedINR — `INR1/inr_modules/models/inr_model.py`

Formula: `Ne = IRI_background + Alpha * Correction`

- `SpatialBasisNet`: Fourier-encoded spatial features `[Lat, Lon, Alt, sin_lt, cos_lt]`
- `TemporalBaseNet`: Fourier-encoded time features
- `VTECSpatialModulator`: Transformer-based FiLM modulation `(gamma, beta)` applied to spatial basis
- `SWPerturbationNet`: Transformer-based additive shift `delta_w` on temporal weights
- `alpha_gate_net`: Dynamic fusion gate (Sigmoid) controlling correction strength
- Heteroscedastic uncertainty: `log_var` estimated from the same basis features

### R-STMRF — `R_STMRF/inr_modules/r_stmrf/r_stmrf_model.py`

Formula: `Ne = IRI_frozen + Decoder(h_spatial, h_temporal_mod)`

Key differences from original INR:
- Uses **SIREN layers** (sine activations with `omega_0=30`) instead of Fourier encoding
- TEC is **not** used in the forward pass — it only constrains gradients via loss
- `GlobalEnvEncoder` (`recurrent_parts.py`): LSTM encoding Kp/F10.7 for additive temporal modulation
- `TecGradientBank` (`tec_gradient_bank.py`): Offline-precomputed TEC gradient directions loaded via memory-map (CPU) or GPU-cached tensor (GPU)
- Physics losses: Chapman vertical smoothness + TEC gradient direction consistency (cosine similarity)
- Physical losses computed every `physics_loss_freq` batches (default 5) to accelerate training
- Uncertainty warmup: first `uncertainty_warmup_epochs` epochs use MSE-only loss, then heteroscedastic loss is enabled

### Physics Losses

**Original INR** (`INR1/inr_modules/losses/physics_losses.py`):
- `heteroscedastic_loss`: NLL with learned variance
- `iri_gradient_direction_loss`: Consistency with IRI background gradient
- `tec_gradient_alignment_loss`: TEC-Ne gradient alignment
- `smoothness_loss_tv`: Total variation smoothness

**R-STMRF** (`R_STMRF/inr_modules/r_stmrf/physics_losses_r_stmrf.py`):
- `chapman_smoothness_loss`: Penalizes 2nd-order altitude derivative (reduces vertical oscillations)
- `tec_gradient_direction_consistency_loss`: Cosine similarity between `grad(Ne)` and TEC gradient direction; only constrains direction, not magnitude

## Key Design Decisions

- **TEC as gradient constraint, not numerical modulation**: TEC is a vertical integral and does not directly correspond to F-region electron density. Enforcing numerical equality is physically incorrect and causes OOM. The R-STMRF v2.0 design uses only gradient direction consistency (weak constraint, weight=0.03).
- **IRI proxy is always frozen**: `requires_grad=False`, but temporarily set to `.train()` mode during forward pass to allow gradient flow through the input coordinates for physics losses.
- **Coordinate system**: Inputs are `(Lat, Lon, Alt, Time)` where Time is hours since `start_date_str`. Altitude range is 120–500 km.
- **Checkpoint output**: Models saved to `./checkpoints/` (INR) or `./checkpoints_r_stmrf/` (R-STMRF). Best model filename: `best_model_vtec_mod.pth` / `best_r_stmrf_model.pth`.

## Memory / Performance Notes

- R-STMRF peak memory at `batch_size=2048`: ~500–700 MB (CPU-compatible)
- On GPU: `batch_size=4096`, AMP enabled, TecGradientBank GPU-cached (~15 MB)
- On CPU: `batch_size=2048`, memory-mapped TEC loading, `physics_loss_freq=20` recommended
- If OOM: reduce `batch_size` → `seq_len` → `basis_dim`/`siren_hidden` → increase `tec_downsample_factor`

## Dependencies

```
torch>=1.9.0
numpy
pandas
matplotlib
scikit-learn
scipy
```
