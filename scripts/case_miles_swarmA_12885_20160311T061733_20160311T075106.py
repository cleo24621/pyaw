"""
生成论文所需的结果
"""

# %% import

import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import spectrogram

from configs import ProjectConfigs
from pyaw.utils import spectral
from pyaw.utils.plot import plot_multi_panel, plot_gridded_panels
from utils import histogram2d, coordinate
from utils.other import (
    OutlierData,
    interpolate_missing,
    align_high2low,
    get_3arrs,
)

# %% basic parameters
window = "hann"
save_dir = r"G:\note\毕业论文\images"

# %% file_paths

data_dir_path = ProjectConfigs.data_dir_path
swarm_type = "A"
file_path_b = os.path.join(
    data_dir_path, "SW_OPER_MAGA_HR_1B_12885_20160311T061733_20160311T075106.pkl"
)
file_path_b_aux = os.path.join(
    data_dir_path, "aux_SW_OPER_MAGA_HR_1B_12885_20160311T061733_20160311T075106.pkl"
)
file_path_b_igrf = os.path.join(
    data_dir_path, "IGRF_SW_OPER_MAGA_HR_1B_12885_20160311T061733_20160311T075106.pkl"
)
file_path_tct16 = os.path.join(
    data_dir_path, "SW_EXPT_EFIA_TCT16_12885_20160311T061733_20160311T075106.pkl"
)
file_path_tct16_aux = os.path.join(
    data_dir_path, "aux_SW_EXPT_EFIA_TCT16_12885_20160311T061733_20160311T075106.pkl"
)

# %% read data as df

df_b = pd.read_pickle(file_path_b)
df_b_aux = pd.read_pickle(file_path_b_aux)
df_b_IGRF = pd.read_pickle(file_path_b_igrf)
df_e = pd.read_pickle(file_path_tct16)
df_e_aux = pd.read_pickle(file_path_tct16_aux)

# %%  --- process data: clip needed data for efficiency

df_b_clip = df_b[["B_NEC", "Longitude", "Latitude", "Radius", "q_NEC_CRF"]]
df_b_aux_clip = df_b_aux[["QDLat", "QDLon", "MLT"]]
df_b_IGRF_clip = df_b_IGRF[["B_NEC_IGRF"]]
df_e_clip = df_e[["Longitude", "Latitude", "Radius", "VsatE", "VsatN", "Ehy", "Ehx"]]
df_e_aux_clip = df_e_aux[["QDLat", "QDLon", "MLT"]]

# %%  --- process data: use time to clip data again

start_time = "20160311T064700"
end_time = "20160311T064900"
df_b_clip = df_b_clip.loc[pd.Timestamp(start_time) : pd.Timestamp(end_time)]
df_b_aux_clip = df_b_aux_clip.loc[pd.Timestamp(start_time) : pd.Timestamp(end_time)]
df_b_IGRF_clip = df_b_IGRF_clip.loc[pd.Timestamp(start_time) : pd.Timestamp(end_time)]
df_e_clip = df_e_clip.loc[pd.Timestamp(start_time) : pd.Timestamp(end_time)]
df_e_aux_clip = df_e_aux_clip.loc[pd.Timestamp(start_time) : pd.Timestamp(end_time)]

latitudes = df_e_clip["Latitude"].values
mlts = df_e_aux_clip["MLT"].values

# %%  --- process data: electric field
Ehx = df_e_clip["Ehx"].values
Ehx_outlier = OutlierData.set_outliers_nan_std(Ehx, 1, print_=True)
Ehx_outlier_interp = interpolate_missing(Ehx_outlier, df_e_clip.index.values)

Ehy = df_e_clip["Ehy"].values
Ehy_outlier = OutlierData.set_outliers_nan_std(Ehy, 1, print_=True)
Ehy_outlier_interp = interpolate_missing(Ehy_outlier, df_e_clip.index.values)

VsatN = df_e_clip["VsatN"].values
VsatE = df_e_clip["VsatE"].values

# %%  --- process data: electric field: sc2nec
rotmat_nec2sc, rotmat_sc2nec = coordinate.NEC2SCandSC2NEC.get_rotmat_nec2sc_sc2nec(
    VsatN, VsatE
)
E_north, E_east = coordinate.NEC2SCandSC2NEC.do_rotation(
    -Ehx_outlier_interp, -Ehy_outlier_interp, rotmat_sc2nec
)  # todo: why need '-'

# %%  ---process data: magnetic field
B_N, B_E, _ = get_3arrs(df_b_clip["B_NEC"].values)
B_N_IGRF, B_E_IGRF, _ = get_3arrs(df_b_IGRF_clip["B_NEC_IGRF"].values)
delta_B_E = B_E - B_E_IGRF
delta_B_N = B_N - B_N_IGRF

# %%  ---process data: magnetic field: downsample, use align time method
datetimes_e = df_e_clip.index.values
datetimes_b = df_b_clip.index.values
delta_B_E_align = align_high2low(delta_B_E, datetimes_b, datetimes_e)
delta_B_N_align = align_high2low(delta_B_N, datetimes_b, datetimes_e)

# %%  ---process data: unify datetimes, i.e., base datetimes
datetimes = datetimes_e

#  ---spectrogram: settings
fs = 16
spectrogram_window_seconds = 4
nperseg = int(spectrogram_window_seconds * fs)

# %%  choose a disturb magnetic field and electric field pair and get spectrogram
frequencies, ts, Sxx_b = spectrogram(
    delta_B_E_align,
    fs=fs,
    window=window,
    nperseg=nperseg,
    mode="complex",
)
_, _, Sxx_e = spectrogram(
    E_north, fs=fs, window=window, nperseg=nperseg, mode="complex"
)

# %% ---spectrogram: get datetime type ndarray for the plot
ts_dt64 = datetimes[0] + [np.timedelta64(int(_), "s") for _ in ts]

# %% ---spectrogram: get the cross spectral
cpsd = Sxx_e * np.conj(Sxx_b)

# %%  --- use new method to get Coherency ---
segment_length_sec = 4  # 越大最后得到的数组的长度越小，取和之前的spectrogram输入的窗口长度是一个不错的选择
try:
    mid_times_all, avg_complex_coh = spectral.calculate_segmented_complex_coherency(
        datetimes,
        delta_B_E_align,
        E_north,
        fs=fs,
        segment_length_sec=segment_length_sec,
        nfft_coh=int(
            fs * segment_length_sec * 0.5
        ),  # Use segment_length_sec/2 second FFT within segments
    )
    # Extract magnitude and phase from the complex result
    avg_coh_magnitude = np.abs(avg_complex_coh)

except ValueError as e:
    print(f"Error calculating Coherency: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# %% preset region parameters for plot annoatations, labels and so on
st_dy = np.datetime64("2016-03-11 06:47:35")
et_dy = np.datetime64("2016-03-11 06:47:55")
st_sta = np.datetime64("2016-03-11 06:47:05")
et_sta = np.datetime64("2016-03-11 06:47:25")

# %%  1st plot: define
subplot_defs = [
    {
        "plot_type": "line",
        "x_data": datetimes,
        "y_data": delta_B_E_align,
        "label": r"$\Delta B_{East}$",
        "title": "East Component of Disturb Magnetic Field",
        "ylabel": "Magnetic Flux Density (nT)",
        "linewidth": 1.8,
    },  # Slightly thicker line
    {
        "plot_type": "line",
        "x_data": datetimes,  # Assuming same time base for E-field line plot
        "y_data": E_north,
        "label": r"$E_{North}$",
        "title": "Northward Component of Electric Field",
        "ylabel": "Electric Field Strength (mV/m)",
        "linewidth": 1.8,
    },
    {
        "plot_type": "pcolormesh",
        "x_data": ts_dt64,
        "y_data": frequencies,
        "z_data": 10 * np.log10(np.abs(Sxx_b)),
        "title": "Power Spectral Density of East Component of Disturb Magnetic Field",
        "ylabel": "Frequency (Hz)",
        "clabel": r"$10log_{10}(S_{bb}(f,t))$",
        "shading": "gouraud",
        # "vmin": -10,
        # "vmax": 10,
        # 'cmap': 'plasma' # Example local override
    },
    {
        "plot_type": "pcolormesh",
        "x_data": ts_dt64,
        "y_data": frequencies,
        "z_data": 10 * np.log10(np.abs(Sxx_e)),
        "title": "Power Spectral Density of Northward Component of Electric Field",
        "ylabel": "Frequency (Hz)",
        "clabel": r"$10log_{10}(S_{EE}(f,t))$",
        "shading": "gouraud",
        # "vmin": -10,
        # "vmax": 10,
    },
    {
        "plot_type": "pcolormesh",
        "x_data": ts_dt64,
        "y_data": frequencies,
        "z_data": 10 * np.log10(np.abs(cpsd)),
        "title": r"The Module of Cross Power Spectral Density of $\Delta B_{East}$ and $E_{North}$",
        "ylabel": "Frequency (Hz)",
        "clabel": r"$10log_{10}(S_{Eb}(f,t))$",
        "shading": "gouraud",
        # "vmin": -10,
        # "vmax": 10,
        # 'vmin': -50 # Example local override for clim
    },
    {
        "plot_type": "line",
        "x_data": mid_times_all,
        "y_data": avg_coh_magnitude,
        "label": r"Coherency",
        "title": "Coherency of $\Delta B_{East}$ and $E_{North}$",
        "ylabel": "Coherency (Unit 1)",
        "linewidth": 1.8,
    },
]

# %% 1st plot: add labels, annotations ...
for subplot_def in subplot_defs[:2]:
    subplot_def["blocks"] = [
        {
            "start": st_sta,
            "end": et_sta,
            "color": "#004488",
            "label": "Static Region",
        },
        {
            "start": st_dy,
            "end": et_dy,
            "color": "#DDAA33",
            "label": "Dynamic Region",
        },
    ]
for subplot_def in subplot_defs[2:-1]:
    subplot_def["vlines"] = {
        "Static Region Start": st_sta,
        "Static Region End": et_sta,
        "Dynamic Region Start": st_dy,
        "Dynamic Region End": et_dy,
    }
subplot_defs[-1]["blocks"] = [
    {
        "start": st_sta,
        "end": et_sta,
        "color": "#004488",
        "label": "Static Region",
    },
    {
        "start": st_dy,
        "end": et_dy,
        "color": "#DDAA33",
        "label": "Dynamic Region",
    },
]
subplot_defs[-1]["hlines"] = [{"y": 0.5, "color": "magenta", "linestyle": "-."}]

# %% --- Define Aux Data for X labels ---
aux_data_for_x = {"Lat": latitudes, "MLT": mlts}

# %% 1st plot: plot settings
# Other options: 'ggplot', 'seaborn-v0_8-talk', 'default'
# See available: print(plt.style.available)
plt.style.use("seaborn-v0_8-paper")  # Good for papers

# %% 1st plot: Call the function to plot
fig, axes = plot_multi_panel(
    subplot_definitions=subplot_defs,
    x_datetime_ref=datetimes,  # Reference times for aux data (note that lats,mlts based on datetimes not tsdt64)
    x_aux_data=aux_data_for_x,  # The aux data arrays
    x_label_step=200,  # Show label every N points
    figsize=(10, 14),
    figure_title=f"Alfven Wave Case: Swarm{swarm_type} from {start_time} to {end_time}",
    # global_cmap=matplotlib.rcParams["image.cmap"],  # default
    global_cmap="viridis",
    global_vmin=-10,  # Example: Manually set global limits
    global_vmax=10,
    use_shared_clims=True,  # Use calculated shared limits unless overridden
    rotate_xticklabels=0,  # Keep horizontal for concise formatter
    # font size
    title_fontsize=11,
    label_fontsize=10,
    tick_label_fontsize=9,
    legend_fontsize=9,
    annotation_fontsize=8,
    panel_label_fontsize=11,  # Control (a), (b)... size
)

# %% 1st plot: save fig with high DPI
save = True
if save:
    output_filename_png = f"1st_plot_Alfven_Wave_Case_Swarm{swarm_type}_from_{start_time}_to_{end_time}.png"
    output_path = os.path.join(save_dir, output_filename_png)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saving figure to {output_filename_png} (300 DPI)")

# %% Region: get clip data
t_mask_dy = (datetimes >= st_dy) & (datetimes <= et_dy)
datetimes_dy = datetimes[t_mask_dy]
delta_B_E_align_dy = delta_B_E_align[t_mask_dy]
E_north_dy = E_north[t_mask_dy]

t_mask_sta = (datetimes >= st_sta) & (datetimes <= et_sta)
datetimes_sta = datetimes[t_mask_sta]
delta_B_E_align_sta = delta_B_E_align[t_mask_sta]
E_north_sta = E_north[t_mask_sta]

# %% Region: get psd
nperseg_psd = 64  # same as the 1st spectrogram nperseg

delta_B_E_align_dy_psd = spectral.PSD(
    delta_B_E_align_dy,
    fs=fs,
    nperseg=nperseg_psd,
    window=window,
    scaling="density",
)  # same arguments setting as spectrogram
E_north_dy_psd = spectral.PSD(
    E_north_dy,
    fs=fs,
    nperseg=nperseg_psd,
    window=window,
    scaling="density",
)
delta_B_E_align_sta_psd = spectral.PSD(
    delta_B_E_align_sta,
    fs=fs,
    nperseg=nperseg_psd,
    window=window,
    scaling="density",
)
E_north_sta_psd = spectral.PSD(
    E_north_sta,
    fs=fs,
    nperseg=nperseg_psd,
    window=window,
    scaling="density",
)

frequencies_psd_dy, Pxx_delta_B_E_align_dy = delta_B_E_align_dy_psd.get_psd()
_, Pxx_E_north_dy = E_north_dy_psd.get_psd()
frequencies_psd_sta, Pxx_delta_B_E_align_sta = delta_B_E_align_sta_psd.get_psd()
_, Pxx_E_north_sta = E_north_sta_psd.get_psd()

# %% Region: cpsd
spectrogram_window_seconds = 4  # compare to the former spectrogram 4
nperseg = int(spectrogram_window_seconds * fs)

frequencies_spec_dy, ts_dy, Sxx_b_dy = spectrogram(
    delta_B_E_align_dy,
    fs=fs,
    window=window,
    nperseg=nperseg,
    mode="complex",
)

_, _, Sxx_e_dy = spectrogram(
    E_north_dy, fs=fs, window=window, nperseg=nperseg, mode="complex"
)

frequencies_spec_sta, ts_sta, Sxx_b_sta = spectrogram(
    delta_B_E_align_sta,
    fs=fs,
    window=window,
    nperseg=nperseg,
    mode="complex",
)
_, _, Sxx_e_sta = spectrogram(
    E_north_sta,
    fs=fs,
    window=window,
    nperseg=nperseg,
    mode="complex",
)

ts_dt64_dy = datetimes_dy[0] + [np.timedelta64(int(_), "s") for _ in ts_dy]
ts_dt64_sta = datetimes_sta[0] + [np.timedelta64(int(_), "s") for _ in ts_sta]

cpsd_dy = Sxx_e_dy * np.conj(Sxx_b_dy)
cpsd_sta = Sxx_e_sta * np.conj(Sxx_b_sta)

# %% Region: phase difference between b and E
cpsd_m_threshold = 0.3
num_bins = 50
cpsd_phase_dy = np.degrees(np.angle(cpsd_dy))
cpsd_m_dy = np.abs(cpsd_dy)
cpsd_phase_dy[cpsd_m_dy < cpsd_m_threshold] = np.nan  # threshold
phase_bins_dy, phase_histogram2d_dy = histogram2d.get_phase_histogram2d(
    frequencies_spec_dy, cpsd_phase_dy, num_bins=num_bins
)

cpsd_phase_sta = np.degrees(np.angle(cpsd_sta))
cpsd_m_sta = np.abs(cpsd_sta)
cpsd_phase_sta[cpsd_m_sta < cpsd_m_threshold] = np.nan  # threshold
phase_bins_sta, phase_histogram2d_sta = histogram2d.get_phase_histogram2d(
    frequencies_spec_sta, cpsd_phase_sta, num_bins=num_bins
)

# %% Region: ratio

eb_ratio_psd_dy = (
    (Pxx_E_north_dy / Pxx_delta_B_E_align_dy) * 1e-3 * 1e9
)  # transform unit
eb_ratio_psd_sta = (Pxx_E_north_sta / Pxx_delta_B_E_align_sta) * 1e-3 * 1e9

# %% Region: lower and upper bound and other parameters
from pyaw.parameters import (
    PhysicalParameters,
    calculate_lower_bound,
    calculate_upper_bound,
    calculate_R,
    calculate_phase_vary_range,
)

mu0 = PhysicalParameters.mu0
Sigma_P_dy = 3.0
va_dy = 1.4e6

boundary_l = calculate_lower_bound(Sigma_P_dy)
boundary_h = calculate_upper_bound(va_dy, Sigma_P_dy)
print(f"boundary_l*mu0: {boundary_l * mu0}")
print(f"boundary_h*mu0: {boundary_h * mu0}")


reflection_coef = calculate_R(v_A=va_dy, Sigma_P=Sigma_P_dy)
print(f"reflection_coef:{reflection_coef}")

phase_vary_range = calculate_phase_vary_range(reflection_coef)
print(f"phase_vary_range_dy: {phase_vary_range}")

# %% 2nd plot: define
nrows, ncols_main = 4, 2
plot_defs = [[None for _ in range(ncols_main)] for _ in range(nrows)]
# Row 0
plot_defs[0][0] = {
    "plot_type": "line",
    "x_data": datetimes_sta,
    "y_data_list": [
        E_north_sta,
        delta_B_E_align_sta,
    ],  # 先用蓝色绘制电场，在用橙色绘制磁场
    "linewidth": 1.8,
    "labels": [r"$E_{North}$", r"$\Delta {B_{East}}$"],
    "title": f"Static Region from\n{st_sta} to {et_sta}",
    "xlabel": "Time (UTC)",
    "ylabel": "Amplitude",
}
plot_defs[0][1] = {
    "plot_type": "line",
    "x_data": datetimes_dy,
    "y_data_list": [E_north_dy, delta_B_E_align_dy],
    "labels": [r"$E_{North}$", r"$\Delta {B_{East}}$"],
    "title": f"Dynamic Region from\n{st_dy} to {et_dy}",
    "xlabel": "Time (UTC)",
    "ylabel": "Amplitude",
}
# Row 1
plot_defs[1][0] = {
    "plot_type": "line",
    "x_data": frequencies_psd_sta,
    "y_data_list": [Pxx_E_north_sta, Pxx_delta_B_E_align_sta],
    "labels": [r"PSD of $E_{North}$", r"PSD of $\Delta {B_{East}}$"],
    "yscale": "log",  # Use log scale
    "title": "PSD of $\Delta {B_{East}}$ and $E_{North}$",
    "xlabel": "Frequency (Hz)",
    "ylabel": "PSD",
}
plot_defs[1][1] = {
    "plot_type": "line",
    "x_data": frequencies_psd_dy,
    "y_data_list": [Pxx_E_north_dy,Pxx_delta_B_E_align_dy ],
    "labels": [r"PSD of $E_{North}$", r"PSD of $\Delta {B_{East}}$"],
    "yscale": "log",  # Use log scale
    "title": "PSD of $\Delta {B_{East}}$ and $E_{North}$",
    "xlabel": "Frequency (Hz)",
    "ylabel": "PSD",
}
# Row 2
plot_defs[2][0] = {
    "plot_type": "line",
    "x_data": frequencies_psd_sta,
    "y_data_list": [eb_ratio_psd_sta],
    "yscale": "log",  # Use log scale
    "labels": [r"ratio $\frac{E_{North}}{\Delta {B_{East}}}$"],
    "title": r"ratio $\frac{E_{North}}{\Delta {B_{East}}}$",
    "xlabel": "Frequency (Hz)",
    "ylabel": "ratio",
    "hlines": [
        {"y": boundary_l},
        {
            "y": boundary_h,
        },
        {"y": va_dy, "label": r"$v_A$"},
    ],
}
plot_defs[2][1] = {
    "plot_type": "line",
    "x_data": frequencies_psd_dy,
    "y_data_list": [eb_ratio_psd_dy],
    "yscale": "log",
    "labels": [r"ratio $\frac{E_{North}}{\Delta {B_{East}}}$"],
    "title": r"ratio $\frac{E_{North}}{\Delta {B_{East}}}$",
    "xlabel": "Frequency (Hz)",
    "ylabel": "ratio",
    "hlines": [
        {"y": boundary_l},
        {
            "y": boundary_h,
        },
        {"y": va_dy, "label": r"$v_A$"},
    ],
}  # Normal line plot here
# Row 3
plot_defs[3][0] = {
    "plot_type": "pcolormesh",
    "x_data": frequencies_spec_sta,
    "y_data": (phase_bins_sta[:-1] + phase_bins_sta[1:]) / 2,
    "z_data": phase_histogram2d_sta.T,
    "title": "The Occurrence of Phase Difference between\n$\Delta {B_{East}}$ and $E_{North}$",
    "xlabel": "Frequency (Hz)",
    "ylabel": "Phase Difference (Degree)",
    "shading": "auto",
    "hlines": [{"y": phase_vary_range[0]}, {"y": phase_vary_range[1]}],
}
plot_defs[3][1] = {
    "plot_type": "pcolormesh",
    "x_data": frequencies_spec_dy,
    "y_data": (phase_bins_dy[:-1] + phase_bins_dy[1:]) / 2,
    "z_data": phase_histogram2d_dy.T,
    "title": "The Occurrence of Phase Difference between\n$\Delta {B_{East}}$ and $E_{North}$",
    "xlabel": "Frequency (Hz)",
    "ylabel": "Phase Difference (Degree)",
    "shading": "auto",
    "hlines": [{"y": phase_vary_range[0]}, {"y": phase_vary_range[1]}],
}
# %% 2nd plot: style
plt.style.use(
    "seaborn-v0_8-paper"
)  # Good for papers. can not set because the former plot already set

# %% 2nd plot: call the function to plot
fig, axes = plot_gridded_panels(
    plot_definitions=plot_defs,
    nrows=4,
    ncols_main=2,
    add_shared_cbar=True,
    shared_cbar_label="Occurrence (Unit 1)",
    figure_title=f"Alfven Wave Case: Swarm{swarm_type} from {start_time} to {end_time}",
    figsize=(10, 16),  # Adjust size
    use_shared_clims=True,  # Use shared clim for spectrograms
    title_fontsize=11,
    label_fontsize=10,
    tick_label_fontsize=9,
    legend_fontsize=9,
    annotation_fontsize=8,
    panel_label_fontsize=11,
)

# %% 2nd plot: save
save = True
if save:
    output_filename_png = f"2nd_plot_Alfven_Wave_Case_Swarm{swarm_type}_from_{start_time}_to_{end_time}.png"
    output_path = os.path.join(save_dir, output_filename_png)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saving figure to {output_filename_png} (300 DPI)")

#%% if show
plt.show()