"""
R-STMRF vs ISR vs IRI Comparison Visualization
==========================================
绘制 Jicamarca 位置 (lat=-11.95°, lon=-76.87°) 的:
- 子图1: ISR 观测的电子密度
- 子图2: R-STMRF 模型预测的电子密度
- 子图3: IRI Proxy 背景场电子密度
- 子图4: R-STMRF - ISR 误差
- 子图5: IRI - ISR 误差
- 独立画布: R-STMRF 预测不确定性 (log_var)

时间范围: 2024-09-05 05:00 UT - 2024-09-06 05:00 UT
高度范围: 120 - 500 km

注意：IDE 可能会报告"无法解析导入"警告，这是正常的，因为模块路径是在运行时动态添加的。
运行时导入将正常工作。
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import h5py
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone, timedelta
from scipy.interpolate import RegularGridInterpolator
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

# Add inr_modules path to sys.path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
inr_module_path = os.path.join(current_dir, 'inr_modules')
if inr_module_path not in sys.path:
    sys.path.insert(0, inr_module_path)

# Verify path exists
if not os.path.exists(inr_module_path):
    raise RuntimeError(f"inr_modules path not found: {inr_module_path}")

# Import R-STMRF components
try:
    from r_stmrf.r_stmrf_model import R_STMRF_Model
    from r_stmrf.tec_gradient_bank import TecGradientBank
    from r_stmrf.sliding_dataset import SlidingWindowBatchProcessor
    from data_managers.space_weather_manager import SpaceWeatherManager
    from data_managers.tec_manager import TECDataManager
    from data_managers.irinc_neural_proxy import IRINeuralProxy
except ImportError as e:
    print(f"Error importing modules. Make sure you're running from the R_STMRF directory.")
    print(f"Current directory: {current_dir}")
    print(f"inr_modules path: {inr_module_path}")
    print(f"sys.path: {sys.path}")
    raise

# ==========================================
# Configuration
# ==========================================
CONFIG = {
    # Data paths
    'iri_proxy_path': r"D:\code11\IRI01\output_results\iri_september_full_proxy.pth",
    'sw_path': r'D:\FYsatellite\EDP_data\kp\OMNI_Kp_F107_20240901_20241001.txt',
    'tec_path': r'D:\IGS\VTEC\tec_map_data.npy',
    'gradient_bank_path': r'D:\IGS\VTEC\tec_gradient_bank.npy',  # 新增：预计算的 TEC 梯度库
    'model_weights': r"D:\code11\IRI01\IRI03\INR1-1\checkpoints_r_stmrf\3-epoch6\best_r_stmrf_model.pth",
    'isr_filepath': r'D:\ISR\DATA\10jicamarca_is_radar(~12°S,低纬磁赤道)\jro20240905_050002.hdf5',

    # Global settings
    'total_hours': 720.0,
    'start_date_str': '2024-09-01 00:00:00',
    'seq_len': 6,

    # Physical ranges
    'lat_range': (-90.0, 90.0),
    'lon_range': (-180.0, 180.0),
    'alt_range': (120.0, 500.0),

    # Target location (Jicamarca)
    'target_lat': -11.95,
    'target_lon': -76.87,

    # Time range for visualization
    'vis_start': '2024-09-05 05:00',
    'vis_end': '2024-09-06 05:00',

    # Altitude range
    'alt_min': 120.0,
    'alt_max': 500.0,
    'alt_resolution': 5.0,  # km

    # Time resolution
    'time_resolution_min': 5,  # minutes

    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    # R-STMRF model config (需要传递给模型初始化)
    'basis_dim': 64,
    'siren_hidden': 128,
    'siren_layers': 3,
    'omega_0': 30.0,
    'env_hidden_dim': 64,

    # Output
    'output_dir': './visualization_outputs'
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)


# ==========================================
# Model Initialization
# ==========================================
def initialize_model(config):
    """Initialize R-STMRF model and managers."""
    device = torch.device(config['device'])
    print(f"Using device: {device}")

    # Space Weather Manager
    print("Initializing Space Weather Manager...")
    sw_manager = SpaceWeatherManager(
        txt_path=config['sw_path'],
        start_date_str=config['start_date_str'],
        total_hours=config['total_hours'],
        seq_len=config['seq_len'],
        device=device
    )

    # TEC Manager (保留用于兼容性，R-STMRF v2.0+ 不使用在线TEC加载)
    print("Initializing TEC Data Manager...")
    tec_manager = TECDataManager(
        tec_map_path=config['tec_path'],
        total_hours=config['total_hours'],
        seq_len=config['seq_len'],
        device=device
    )

    # TEC Gradient Bank (新架构：离线预计算梯度)
    print("Initializing TEC Gradient Bank (offline pre-computed)...")
    gradient_bank = TecGradientBank(
        gradient_bank_path=config['gradient_bank_path'],
        total_hours=config['total_hours'],
        device=device
    )

    # IRI Neural Proxy
    print("Loading IRI Neural Proxy...")
    iri_proxy = IRINeuralProxy(layers=[4, 128, 128, 128, 128, 1]).to(device)
    iri_state = torch.load(config['iri_proxy_path'], map_location=device)
    iri_proxy.load_state_dict(iri_state)
    iri_proxy.eval()

    # R-STMRF Model
    print("Initializing R-STMRF Model...")
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

    # Load trained weights
    print("Loading trained R-STMRF weights...")
    model_state = torch.load(config['model_weights'], map_location=device)
    model.load_state_dict(model_state)
    model.eval()
    print("Model loaded successfully.")

    # Create batch processor
    batch_processor = SlidingWindowBatchProcessor(sw_manager, tec_manager, device)

    return model, sw_manager, tec_manager, gradient_bank, batch_processor, device


# ==========================================
# R-STMRF and IRI Inference for Time-Altitude Profile
# ==========================================
def run_r_stmrf_iri_profile(model, batch_processor, gradient_bank, config, device):
    """
    Run R-STMRF and IRI inference for a single location over time and altitude.
    Returns: timestamps (datetime), altitudes (km), R-STMRF log10(Ne), IRI log10(Ne), log_var
    """
    lat = config['target_lat']
    lon = config['target_lon']

    # Create time array
    start_time = pd.to_datetime(config['vis_start'])
    end_time = pd.to_datetime(config['vis_end'])
    ref_time = pd.to_datetime(config['start_date_str'])

    time_delta = timedelta(minutes=config['time_resolution_min'])
    timestamps = []
    current = start_time
    while current <= end_time:
        timestamps.append(current)
        current += time_delta
    timestamps = np.array(timestamps)
    n_times = len(timestamps)

    # Convert to hours since reference
    time_hours = np.array([(t - ref_time).total_seconds() / 3600.0 for t in timestamps], dtype=np.float32)

    # Create altitude array
    altitudes = np.arange(config['alt_min'], config['alt_max'] + config['alt_resolution'],
                          config['alt_resolution'], dtype=np.float32)
    n_alts = len(altitudes)

    print(f"R-STMRF/IRI Inference Grid:")
    print(f"  Location: lat={lat}°, lon={lon}°")
    print(f"  Time points: {n_times} ({config['vis_start']} to {config['vis_end']})")
    print(f"  Altitude levels: {n_alts} ({config['alt_min']}-{config['alt_max']} km)")

    # Create full coordinate grid
    TIME, ALT = np.meshgrid(time_hours, altitudes)  # shape: (n_alts, n_times)

    lat_flat = np.full(TIME.size, lat, dtype=np.float32)
    lon_flat = np.full(TIME.size, lon, dtype=np.float32)
    alt_flat = ALT.flatten().astype(np.float32)
    time_flat = TIME.flatten().astype(np.float32)

    coords = np.stack([lat_flat, lon_flat, alt_flat, time_flat], axis=1)
    coords_tensor = torch.from_numpy(coords).to(device)

    batch_size = coords_tensor.shape[0]

    # Process in batches to avoid memory issues
    batch_limit = 10000
    r_stmrf_results = []
    iri_results = []
    log_var_results = []

    print(f"  Running inference ({batch_size} total points)...")

    for i in range(0, batch_size, batch_limit):
        end_idx = min(i + batch_limit, batch_size)
        batch_coords = coords_tensor[i:end_idx]
        batch_n = batch_coords.shape[0]

        # Get SW sequence for this batch
        batch_times = batch_coords[:, 3]

        # R-STMRF uses per-point SW sequence query
        sw_seq = batch_processor.sw_manager.get_drivers_sequence(batch_times)  # [Batch, Seq, 2]

        # Get TEC gradient direction from pre-computed bank
        tec_grad_direction = gradient_bank.get_interpolated_gradient(batch_times)  # [Batch, 2, H, W]

        # Run model
        with torch.no_grad():
            # Get IRI background
            iri_bg = model.get_background(
                batch_coords[:, 0],
                batch_coords[:, 1],
                batch_coords[:, 2],
                batch_coords[:, 3]
            ).squeeze()

            # Get R-STMRF prediction (4 return values)
            r_stmrf_pred, log_var, correction, extras = model(batch_coords, sw_seq, tec_grad_direction)
            r_stmrf_pred = r_stmrf_pred.squeeze()
            log_var = log_var.squeeze()

            r_stmrf_results.append(r_stmrf_pred.cpu().numpy())
            iri_results.append(iri_bg.cpu().numpy())
            log_var_results.append(log_var.cpu().numpy())

    # Combine results
    r_stmrf_ne = np.concatenate(r_stmrf_results)
    iri_ne = np.concatenate(iri_results)
    log_var_ne = np.concatenate(log_var_results)

    r_stmrf_ne_2d = r_stmrf_ne.reshape(n_alts, n_times)  # shape: (alt, time)
    iri_ne_2d = iri_ne.reshape(n_alts, n_times)
    log_var_ne_2d = log_var_ne.reshape(n_alts, n_times)

    print(f"  R-STMRF log10(Ne) range: {np.nanmin(r_stmrf_ne_2d):.2f} - {np.nanmax(r_stmrf_ne_2d):.2f}")
    print(f"  IRI log10(Ne) range: {np.nanmin(iri_ne_2d):.2f} - {np.nanmax(iri_ne_2d):.2f}")
    print(f"  log_var range: {np.nanmin(log_var_ne_2d):.2f} - {np.nanmax(log_var_ne_2d):.2f}")

    return timestamps, altitudes, r_stmrf_ne_2d, iri_ne_2d, log_var_ne_2d


# ==========================================
# Read ISR Data
# ==========================================
def read_isr_data(filepath, config):
    """
    Read and preprocess ISR HDF5 data.
    Filter to specified time and altitude range.
    """
    print(f"\nReading ISR data from: {filepath}")

    data = {}

    with h5py.File(filepath, 'r') as f:
        # Read timestamps
        timestamps_unix = f['/Data/Array Layout/timestamps'][:]
        timestamps = np.array([
            datetime.fromtimestamp(ts, tz=timezone.utc)
            for ts in timestamps_unix
        ])

        # Read altitudes
        gdalt = f['/Data/Array Layout/gdalt'][:]

        # Read Ne
        ne = f['/Data/Array Layout/2D Parameters/ne'][:]
        dne = f['/Data/Array Layout/2D Parameters/dne'][:]

    # Quality control
    ne[ne <= 0] = np.nan
    ne[ne > 1e15] = np.nan
    dne[dne <= 0] = np.nan
    relative_error = np.abs(dne / ne)
    ne[relative_error > 1.0] = np.nan

    # Convert to log10
    log_ne = np.log10(ne)

    # Filter time range
    vis_start = pd.to_datetime(config['vis_start']).replace(tzinfo=timezone.utc)
    vis_end = pd.to_datetime(config['vis_end']).replace(tzinfo=timezone.utc)

    time_mask = (timestamps >= vis_start) & (timestamps <= vis_end)

    # Filter altitude range
    alt_mask = (gdalt >= config['alt_min']) & (gdalt <= config['alt_max'])

    data['timestamps'] = timestamps[time_mask]
    data['gdalt'] = gdalt[alt_mask]
    data['log_ne'] = log_ne[np.ix_(alt_mask, time_mask)]  # (alt, time)

    print(f"  ISR time range: {data['timestamps'][0]} - {data['timestamps'][-1]}")
    print(f"  ISR altitude range: {data['gdalt'].min():.1f} - {data['gdalt'].max():.1f} km")
    print(f"  ISR data shape: {data['log_ne'].shape}")
    print(f"  ISR log10(Ne) range: {np.nanmin(data['log_ne']):.2f} - {np.nanmax(data['log_ne']):.2f}")

    return data


# ==========================================
# Interpolate to ISR Grid
# ==========================================
def interpolate_to_isr_grid(model_timestamps, model_altitudes, model_ne, isr_data):
    """
    Interpolate model data (R-STMRF or IRI) to ISR time-altitude grid.
    """
    # Convert model timestamps to numeric (hours since first timestamp)
    model_times_num = np.array([(t - model_timestamps[0]).total_seconds() / 3600.0
                                for t in model_timestamps])

    # Convert ISR timestamps to same reference
    isr_times_num = np.array([
        (t.replace(tzinfo=None) - model_timestamps[0].to_pydatetime().replace(tzinfo=None)).total_seconds() / 3600.0
        for t in isr_data['timestamps']
    ])

    # Create interpolator (alt, time) -> value
    interpolator = RegularGridInterpolator(
        (model_altitudes, model_times_num),
        model_ne,
        method='linear',
        bounds_error=False,
        fill_value=np.nan
    )

    # Create ISR grid points
    ISR_ALT, ISR_TIME = np.meshgrid(isr_data['gdalt'], isr_times_num, indexing='ij')
    points = np.stack([ISR_ALT.flatten(), ISR_TIME.flatten()], axis=1)

    # Interpolate
    model_on_isr_grid = interpolator(points).reshape(ISR_ALT.shape)

    return model_on_isr_grid


# ==========================================
# 5-Panel Plotting Function
# ==========================================
def plot_5panel_comparison(r_stmrf_timestamps, r_stmrf_altitudes, r_stmrf_ne, iri_ne,
                           isr_data, r_stmrf_on_isr, iri_on_isr, config):
    """
    Create 5-panel comparison figure:
    - Panel 1: ISR Ne (观测)
    - Panel 2: R-STMRF Ne (模型预测)
    - Panel 3: IRI Proxy Ne (背景场)
    - Panel 4: R-STMRF - ISR 误差
    - Panel 5: IRI - ISR 误差
    """

    fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)
    plt.subplots_adjust(hspace=0.12, right=0.88)

    # =============================================
    # Create mesh grids
    # =============================================
    # R-STMRF/IRI mesh
    model_times_num = mdates.date2num(r_stmrf_timestamps)
    dt = np.diff(model_times_num)
    time_edges_model = np.concatenate([
        [model_times_num[0] - dt[0]/2],
        model_times_num[:-1] + dt/2,
        [model_times_num[-1] + dt[-1]/2]
    ])

    dh = np.diff(r_stmrf_altitudes)
    alt_edges_model = np.concatenate([
        [r_stmrf_altitudes[0] - dh[0]/2],
        r_stmrf_altitudes[:-1] + dh/2,
        [r_stmrf_altitudes[-1] + dh[-1]/2]
    ])

    T_model, H_model = np.meshgrid(time_edges_model, alt_edges_model)

    # ISR mesh
    isr_times_num = mdates.date2num(isr_data['timestamps'])
    dt_isr = np.diff(isr_times_num)
    time_edges_isr = np.concatenate([
        [isr_times_num[0] - dt_isr[0]/2],
        isr_times_num[:-1] + dt_isr/2,
        [isr_times_num[-1] + dt_isr[-1]/2]
    ])

    dh_isr = np.diff(isr_data['gdalt'])
    alt_edges_isr = np.concatenate([
        [isr_data['gdalt'][0] - dh_isr[0]/2],
        isr_data['gdalt'][:-1] + dh_isr/2,
        [isr_data['gdalt'][-1] + dh_isr[-1]/2]
    ])

    T_isr, H_isr = np.meshgrid(time_edges_isr, alt_edges_isr)

    # =============================================
    # Unified color scales
    # =============================================
    # Ne color scale (for panels 1-3)
    vmin_ne = min(np.nanmin(r_stmrf_ne), np.nanmin(iri_ne), np.nanmin(isr_data['log_ne']))
    vmax_ne = max(np.nanmax(r_stmrf_ne), np.nanmax(iri_ne), np.nanmax(isr_data['log_ne']))

    # Difference color scale (for panels 4-5)
    diff_r_stmrf = r_stmrf_on_isr - isr_data['log_ne']
    diff_iri = iri_on_isr - isr_data['log_ne']
    vmax_diff = max(np.nanmax(np.abs(diff_r_stmrf)), np.nanmax(np.abs(diff_iri)))
    vmax_diff = min(vmax_diff, 2.0)  # Cap at ±2

    # =============================================
    # Panel 1: ISR Ne (观测)
    # =============================================
    ax1 = axes[0]
    pcm1 = ax1.pcolormesh(T_isr, H_isr, isr_data['log_ne'],
                          cmap='jet', vmin=vmin_ne, vmax=vmax_ne, shading='flat')
    cbar1 = fig.colorbar(pcm1, ax=ax1, pad=0.02, aspect=15)
    cbar1.set_label(r'$\log_{10}(N_e)$ [m$^{-3}$]', fontsize=11)

    ax1.set_ylabel('Altitude (km)', fontsize=11)
    ax1.set_title('(a) Jicamarca ISR Observation', fontsize=12, fontweight='bold', loc='left')
    ax1.set_ylim([config['alt_min'], config['alt_max']])
    ax1.grid(True, alpha=0.3, linestyle='--')

    # =============================================
    # Panel 2: R-STMRF Ne (模型预测)
    # =============================================
    ax2 = axes[1]
    pcm2 = ax2.pcolormesh(T_model, H_model, r_stmrf_ne,
                          cmap='jet', vmin=vmin_ne, vmax=vmax_ne, shading='flat')
    cbar2 = fig.colorbar(pcm2, ax=ax2, pad=0.02, aspect=15)
    cbar2.set_label(r'$\log_{10}(N_e)$ [m$^{-3}$]', fontsize=11)

    ax2.set_ylabel('Altitude (km)', fontsize=11)
    ax2.set_title('(b) R-STMRF Model Prediction', fontsize=12, fontweight='bold', loc='left')
    ax2.set_ylim([config['alt_min'], config['alt_max']])
    ax2.grid(True, alpha=0.3, linestyle='--')

    # =============================================
    # Panel 3: IRI Proxy Ne (背景场)
    # =============================================
    ax3 = axes[2]
    pcm3 = ax3.pcolormesh(T_model, H_model, iri_ne,
                          cmap='jet', vmin=vmin_ne, vmax=vmax_ne, shading='flat')
    cbar3 = fig.colorbar(pcm3, ax=ax3, pad=0.02, aspect=15)
    cbar3.set_label(r'$\log_{10}(N_e)$ [m$^{-3}$]', fontsize=11)

    ax3.set_ylabel('Altitude (km)', fontsize=11)
    ax3.set_title('(c) IRI-2020 Background (Neural Proxy)', fontsize=12, fontweight='bold', loc='left')
    ax3.set_ylim([config['alt_min'], config['alt_max']])
    ax3.grid(True, alpha=0.3, linestyle='--')

    # =============================================
    # Panel 4: R-STMRF - ISR 误差
    # =============================================
    ax4 = axes[3]
    pcm4 = ax4.pcolormesh(T_isr, H_isr, diff_r_stmrf,
                          cmap='bwr', vmin=-vmax_diff, vmax=vmax_diff, shading='flat')
    cbar4 = fig.colorbar(pcm4, ax=ax4, pad=0.02, aspect=15)
    cbar4.set_label(r'$\Delta \log_{10}(N_e)$', fontsize=11)

    ax4.set_ylabel('Altitude (km)', fontsize=11)
    ax4.set_title('(d) Difference: R-STMRF − ISR', fontsize=12, fontweight='bold', loc='left')
    ax4.set_ylim([config['alt_min'], config['alt_max']])
    ax4.grid(True, alpha=0.3, linestyle='--')

    # =============================================
    # Panel 5: IRI - ISR 误差
    # =============================================
    ax5 = axes[4]
    pcm5 = ax5.pcolormesh(T_isr, H_isr, diff_iri,
                          cmap='bwr', vmin=-vmax_diff, vmax=vmax_diff, shading='flat')
    cbar5 = fig.colorbar(pcm5, ax=ax5, pad=0.02, aspect=15)
    cbar5.set_label(r'$\Delta \log_{10}(N_e)$', fontsize=11)

    ax5.set_xlabel('Time (UT)', fontsize=11)
    ax5.set_ylabel('Altitude (km)', fontsize=11)
    ax5.set_title('(e) Difference: IRI − ISR', fontsize=12, fontweight='bold', loc='left')
    ax5.set_ylim([config['alt_min'], config['alt_max']])
    ax5.grid(True, alpha=0.3, linestyle='--')

    # Time axis formatting
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax5.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax5.xaxis.set_minor_locator(mdates.MinuteLocator(interval=30))

    # =============================================
    # Super title
    # =============================================
    date_str = r_stmrf_timestamps[0].strftime('%Y-%m-%d')
    fig.suptitle(
        f'R-STMRF vs IRI vs ISR Comparison - Jicamarca ({config["target_lat"]}°N, {config["target_lon"]}°E)\n{date_str}',
        fontsize=14, fontweight='bold', y=0.995
    )

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    # =============================================
    # Calculate and print statistics
    # =============================================
    print("\n" + "=" * 60)
    print("Statistics (on ISR grid):")
    print("=" * 60)

    valid_mask_r_stmrf = ~np.isnan(diff_r_stmrf)
    valid_mask_iri = ~np.isnan(diff_iri)

    if np.any(valid_mask_r_stmrf):
        rmse_r_stmrf = np.sqrt(np.nanmean(diff_r_stmrf**2))
        bias_r_stmrf = np.nanmean(diff_r_stmrf)
        mae_r_stmrf = np.nanmean(np.abs(diff_r_stmrf))
        print(f"  R-STMRF vs ISR:")
        print(f"    RMSE: {rmse_r_stmrf:.4f}")
        print(f"    Bias: {bias_r_stmrf:.4f}")
        print(f"    MAE:  {mae_r_stmrf:.4f}")

    if np.any(valid_mask_iri):
        rmse_iri = np.sqrt(np.nanmean(diff_iri**2))
        bias_iri = np.nanmean(diff_iri)
        mae_iri = np.nanmean(np.abs(diff_iri))
        print(f"  IRI vs ISR:")
        print(f"    RMSE: {rmse_iri:.4f}")
        print(f"    Bias: {bias_iri:.4f}")
        print(f"    MAE:  {mae_iri:.4f}")

    if np.any(valid_mask_r_stmrf) and np.any(valid_mask_iri):
        print(f"\n  Improvement (R-STMRF over IRI):")
        print(f"    RMSE reduction: {(1 - rmse_r_stmrf/rmse_iri)*100:.1f}%")
        print(f"    MAE reduction:  {(1 - mae_r_stmrf/mae_iri)*100:.1f}%")

    print("=" * 60)

    # =============================================
    # Save figure
    # =============================================
    output_path = os.path.join(config['output_dir'], 'r_stmrf_iri_isr_5panel_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nFigure saved to: {output_path}")

    plt.show()

    return fig


# ==========================================
# Uncertainty Visualization (Separate Canvas)
# ==========================================
def plot_uncertainty(r_stmrf_timestamps, r_stmrf_altitudes, log_var_ne, config):
    """
    Create separate figure for R-STMRF prediction uncertainty (log_var).
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    # Create mesh grid
    model_times_num = mdates.date2num(r_stmrf_timestamps)
    dt = np.diff(model_times_num)
    time_edges = np.concatenate([
        [model_times_num[0] - dt[0]/2],
        model_times_num[:-1] + dt/2,
        [model_times_num[-1] + dt[-1]/2]
    ])

    dh = np.diff(r_stmrf_altitudes)
    alt_edges = np.concatenate([
        [r_stmrf_altitudes[0] - dh[0]/2],
        r_stmrf_altitudes[:-1] + dh/2,
        [r_stmrf_altitudes[-1] + dh[-1]/2]
    ])

    T, H = np.meshgrid(time_edges, alt_edges)

    # Plot log_var
    pcm = ax.pcolormesh(T, H, log_var_ne, cmap='viridis', shading='flat')
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02, aspect=15)
    cbar.set_label(r'$\log(\sigma^2)$ (Log Variance)', fontsize=11)

    ax.set_xlabel('Time (UT)', fontsize=11)
    ax.set_ylabel('Altitude (km)', fontsize=11)
    ax.set_title('R-STMRF Prediction Uncertainty (Heteroscedastic Variance)',
                 fontsize=12, fontweight='bold')
    ax.set_ylim([config['alt_min'], config['alt_max']])
    ax.grid(True, alpha=0.3, linestyle='--')

    # Time axis formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=30))

    # Super title
    date_str = r_stmrf_timestamps[0].strftime('%Y-%m-%d')
    fig.suptitle(
        f'R-STMRF Uncertainty Estimation - Jicamarca ({config["target_lat"]}°N, {config["target_lon"]}°E)\n{date_str}',
        fontsize=14, fontweight='bold'
    )

    plt.tight_layout()

    # =============================================
    # Statistics
    # =============================================
    print("\n" + "=" * 60)
    print("Uncertainty Statistics:")
    print("=" * 60)
    print(f"  log_var range: [{np.nanmin(log_var_ne):.2f}, {np.nanmax(log_var_ne):.2f}]")
    print(f"  log_var mean: {np.nanmean(log_var_ne):.2f}")
    print(f"  log_var std: {np.nanstd(log_var_ne):.2f}")

    # Convert to standard deviation
    sigma = np.sqrt(np.exp(log_var_ne))
    print(f"\n  Corresponding σ (standard deviation) range:")
    print(f"    σ range: [{np.nanmin(sigma):.4f}, {np.nanmax(sigma):.4f}] log10(Ne)")
    print(f"    σ mean: {np.nanmean(sigma):.4f} log10(Ne)")
    print("=" * 60)

    # =============================================
    # Save figure
    # =============================================
    output_path = os.path.join(config['output_dir'], 'r_stmrf_uncertainty.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nUncertainty figure saved to: {output_path}")

    plt.show()

    return fig


# ==========================================
# Main
# ==========================================
def main():
    print("=" * 60)
    print("  R-STMRF vs IRI vs ISR Comparison")
    print("  Location: Jicamarca (-11.95°N, -76.87°E)")
    print("=" * 60)

    # 1. Initialize model
    model, sw_manager, tec_manager, gradient_bank, batch_processor, device = initialize_model(CONFIG)

    # 2. Run R-STMRF and IRI inference
    print("\n" + "=" * 60)
    print("Running R-STMRF and IRI inference...")
    r_stmrf_timestamps, r_stmrf_altitudes, r_stmrf_ne, iri_ne, log_var_ne = run_r_stmrf_iri_profile(
        model, batch_processor, gradient_bank, CONFIG, device
    )

    # 3. Read ISR data
    print("\n" + "=" * 60)
    isr_data = read_isr_data(CONFIG['isr_filepath'], CONFIG)

    # 4. Interpolate R-STMRF and IRI to ISR grid
    print("\n" + "=" * 60)
    print("Interpolating models to ISR grid...")
    r_stmrf_on_isr = interpolate_to_isr_grid(r_stmrf_timestamps, r_stmrf_altitudes, r_stmrf_ne, isr_data)
    iri_on_isr = interpolate_to_isr_grid(r_stmrf_timestamps, r_stmrf_altitudes, iri_ne, isr_data)
    print("  Interpolation complete.")

    # 5. Plot 5-panel comparison
    print("\n" + "=" * 60)
    print("Generating 5-panel comparison figure...")
    fig_comparison = plot_5panel_comparison(
        r_stmrf_timestamps, r_stmrf_altitudes, r_stmrf_ne, iri_ne,
        isr_data, r_stmrf_on_isr, iri_on_isr, CONFIG
    )

    # 6. Plot uncertainty (separate canvas)
    print("\n" + "=" * 60)
    print("Generating uncertainty visualization...")
    fig_uncertainty = plot_uncertainty(
        r_stmrf_timestamps, r_stmrf_altitudes, log_var_ne, CONFIG
    )

    print("\n" + "=" * 60)
    print("  Visualization Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
