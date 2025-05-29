# %% import
import os.path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

from configs import ProjectConfigs

import pyaw.utils
from core import dmsp
from utils import spectral
from src.pyaw import plot_multi_panel, plot_gridded_panels
from utils import histogram2d


#%% save or not. show or not.
save = True
show = False

# %% basic parameters
fs = 1
window = "hann"
save_dir = r"G:\master\pyaw\scripts\results\aw_cases\dmsp"

# %% file paths
data_dir_path = ProjectConfigs.data_dir_path
dmsp_number = "F18"
file_name_s3 = "dmsp-f18_ssies-3_thermal-plasma_201401010124_v01.cdf"  # 一轨
file_name_ssm = "dmsp-f18_ssm_magnetometer_20140101_v1.0.4.cdf"  # 1天
file_path_ssies3 = os.path.join(data_dir_path, file_name_s3)
file_path_ssm = os.path.join(data_dir_path, file_name_ssm)

# %% read data as df
ssies3_ssm = dmsp.SPDF.SSIES3CoupleSSM(file_path_ssies3, file_path_ssm)
df_whole = ssies3_ssm.ssies3_ssm_df

# %% according to lat, clip df
df = df_whole[df_whole["glat"] > 50].copy()

# %% set base datetimes, lat, MLT(of AACGM)
datetimes = df.index.values
dt = datetime.fromisoformat(str(datetimes[0]))
start_time = dt.strftime("%Y%m%dT%H%M%S")
dt = datetime.fromisoformat(str(datetimes[-1]))
end_time = dt.strftime("%Y%m%dT%H%M%S")

latitudes = df["glat"].values
mlts = df[
    "sc_aacgm_ltime"
].values  # note that all column names are lowercase. also get apex

# %% choose a disturb magnetic field and electric field pair. For DMSP, because the quality control sets nan values to
# ion velocity, and the E is calculating from ion velocity and magnetic field, so for enough data points, choose y
# component of b in S/C coordinate system, and choose x component of E in S/C coordinate system.
# and fill nan values use ffill and bfill, not use linear.
b_sc2 = (
    df["b1_s3_sc2"].copy().ffill().bfill().values
)  # not use linear, because too many nan will cause large values
E_sc1 = df["E_s3_sc1"].copy().ffill().bfill().values
assert not np.isnan(b_sc2).any(), "have nan values"
assert not np.isnan(E_sc1).any(), "have nan values"

# %% spectrogram settings
spectrogram_window_seconds = 4
nperseg = int(spectrogram_window_seconds * fs)

# %% get spectrogram
frequencies, ts, Sxx_b = spectrogram(
    b_sc2,
    fs=fs,
    window=window,
    nperseg=nperseg,
    mode="complex",
)
_, _, Sxx_e = spectrogram(E_sc1, fs=fs, window=window, nperseg=nperseg, mode="complex")

# %% spectrogram: get datetime type ndarray for the plot
ts_dt64 = datetimes[0] + [np.timedelta64(int(_), "s") for _ in ts]

# %% get the cross spectral
cpsd = Sxx_e * np.conj(Sxx_b)

# %% get Coherency
segment_length_sec = 10  # 越大最后得到的数组的长度越小
try:
    mid_times_all, avg_complex_coh = spectral.calculate_segmented_complex_coherency(
        datetimes,
        b_sc2,
        E_sc1,
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
st_dy = np.datetime64("2014-01-01T01:49:00")
et_dy = np.datetime64("2014-01-01T01:51:00")
st_sta = np.datetime64("2014-01-01T01:52:00")  # modify
et_sta = np.datetime64("2014-01-01T01:54:00")
# st_sta = np.datetime64("2014-01-01T01:54:00")
# et_sta = np.datetime64("2014-01-01T01:56:00")
# et_sta = np.datetime64("2014-01-01T01:58:00")
# %% 1st plot: define

subplot_defs = [
    {
        "plot_type": "line",
        "x_data": datetimes,
        "y_data": b_sc2,
        "label": r"$\Delta B_{y}$",
        "title": "y Component of Disturb Magnetic Field",
        "ylabel": "Magnetic Flux Density (nT)",
        "linewidth": 1.8,
    },  # Slightly thicker line
    {
        "plot_type": "line",
        "x_data": datetimes,  # Assuming same time base for E-field line plot
        "y_data": E_sc1,
        "label": r"$E_{x}$",
        "title": "x Component of Electric Field",
        "ylabel": "Electric Field Strength (mV/m)",
        "linewidth": 1.8,
    },
    {
        "plot_type": "pcolormesh",
        "x_data": ts_dt64,
        "y_data": frequencies,
        "z_data": np.abs(Sxx_b),
        "title": "Power Spectral Density of y Component of Disturb Magnetic Field",
        "ylabel": "Frequency (Hz)",
        "clabel": r"$S_{bb}(f,t)$",
        "shading": "gouraud",
    },
    {
        "plot_type": "pcolormesh",
        "x_data": ts_dt64,
        "y_data": frequencies,
        "z_data": np.abs(Sxx_e),
        "title": "Power Spectral Density of x Component of Electric Field",
        "ylabel": "Frequency (Hz)",
        "clabel": r"$S_{EE}(f,t)$",
        "shading": "gouraud",
    },
    {
        "plot_type": "pcolormesh",
        "x_data": ts_dt64,
        "y_data": frequencies,
        "z_data": np.abs(cpsd),
        "title": r"The Module of Cross Power Spectral Density of $b_y$ and $E_{x}$",
        "ylabel": "Frequency (Hz)",
        "clabel": r"$S_{Eb}(f,t)$",
        "shading": "gouraud",
    },
    {
        "plot_type": "line",
        "x_data": mid_times_all,
        "y_data": avg_coh_magnitude,
        "label": r"Coherency",
        "title": "Coherency of $\Delta B_{y}$ and $E_{x}$",
        "ylabel": "Coherency (Unit 1)",
        "linewidth": 1.8,
    },  # the coherency subfigure
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

# %% 1st plot: Define Aux Data for X labels
aux_data_for_x = {"Lat": latitudes, "MLT": mlts}

# %% 1st plot: plot settings
plt.style.use("seaborn-v0_8-paper")  # Good for papers

# %% 1st plot: Call the function to plot
fig, axes = plot_multi_panel(
    subplot_definitions=subplot_defs,
    # Reference times for aux data (note that lats,mlts based on datetimes not
    # tsdt64)
    x_datetime_ref=datetimes,
    x_aux_data=aux_data_for_x,  # The aux data arrays
    x_label_step=100,  # Show label every N points
    figsize=(10, 14),
    figure_title=f"Alfven Wave Case: DMSP {dmsp_number} from {start_time} to {end_time}",
    global_cmap="viridis",
    global_vmin=-20,
    global_vmax=20,
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
if save:
    output_filename_png = f"1st_plot_Alfven_Wave_Case_DMSP_{dmsp_number}_from_{start_time}_to_{end_time}.png"
    output_path = os.path.join(save_dir, output_filename_png)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saving figure to {output_filename_png} (300 DPI)")

# %% Region: get clip data
t_mask_dy = (datetimes >= st_dy) & (datetimes <= et_dy)
datetimes_dy = datetimes[t_mask_dy]
b_sc2_dy = b_sc2[t_mask_dy]
E_sc1_dy = E_sc1[t_mask_dy]

t_mask_sta = (datetimes >= st_sta) & (datetimes <= et_sta)
datetimes_sta = datetimes[t_mask_sta]
b_sc2_sta = b_sc2[t_mask_sta]
E_sc1_sta = E_sc1[t_mask_sta]

# %% Region: get psd
nperseg_psd = 64

b_sc2_dy_psd = spectral.PSD(
    b_sc2_dy,
    fs=fs,
    nperseg=nperseg_psd,
    window=window,
    scaling="density",
)
E_sc1_dy_psd = spectral.PSD(
    E_sc1_dy,
    fs=fs,
    nperseg=nperseg_psd,
    window=window,
    scaling="density",
)
b_sc2_sta_psd = spectral.PSD(
    b_sc2_sta,
    fs=fs,
    nperseg=nperseg_psd,
    window=window,
    scaling="density",
)
E_sc1_sta_psd = spectral.PSD(
    E_sc1_sta,
    fs=fs,
    nperseg=nperseg_psd,
    window=window,
    scaling="density",
)

frequencies_psd_dy, Pxx_b_sc2_dy = b_sc2_dy_psd.get_psd()
_, Pxx_E_sc1_dy = E_sc1_dy_psd.get_psd()

frequencies_psd_sta, Pxx_b_sc2_sta = b_sc2_sta_psd.get_psd()
_, Pxx_E_sc1_sta = E_sc1_sta_psd.get_psd()

# %% Region: cpsd
spectrogram_window_seconds = 16  # compare to the former spectrogram 4
nperseg = int(spectrogram_window_seconds * fs)

frequencies_spec_dy, ts_dy, Sxx_b_dy = spectrogram(
    b_sc2_dy,
    fs=fs,
    window=window,
    nperseg=nperseg,
    mode="complex",
)
_, _, Sxx_e_dy = spectrogram(
    E_sc1_dy, fs=fs, window=window, nperseg=nperseg, mode="complex"
)
frequencies_spec_sta, ts_sta, Sxx_b_sta = spectrogram(
    b_sc2_sta,
    fs=fs,
    window=window,
    nperseg=nperseg,
    mode="complex",
)
_, _, Sxx_e_sta = spectrogram(
    E_sc1_sta,
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
cpsd_phase_dy[cpsd_m_dy < cpsd_m_threshold] = np.nan
phase_bins_dy, phase_histogram2d_dy = pyaw.utils.get_phase_histogram2d(
    frequencies_spec_dy, cpsd_phase_dy, num_bins=num_bins
)

cpsd_phase_sta = np.degrees(np.angle(cpsd_sta))
cpsd_m_sta = np.abs(cpsd_sta)
cpsd_phase_sta[cpsd_m_sta < cpsd_m_threshold] = np.nan
phase_bins_sta, phase_histogram2d_sta = pyaw.utils.get_phase_histogram2d(
    frequencies_spec_sta, cpsd_phase_sta, num_bins=num_bins
)

# %% Region: ratio
eb_ratio_psd_dy = (Pxx_E_sc1_dy / Pxx_b_sc2_dy) * 1e-3 * 1e9  # transform unit
eb_ratio_psd_sta = (Pxx_E_sc1_sta / Pxx_b_sc2_sta) * 1e-3 * 1e9

# %% Region: lower and upper bound and other parameters
from src.pyaw import VACUUM_PERMEABILITY, Alfven

pedersen_conductance_dynamic = 3.0
alfven_velocity_dynamic = 1.4e6

alfven = Alfven()

boundary_l = alfven.calculate_lower_boundary(pedersen_conductance=pedersen_conductance_dynamic)
boundary_h = alfven.calculate_upper_boundary(alfven_velocity=alfven_velocity_dynamic, pedersen_conductance=pedersen_conductance_dynamic)
print(f"boundary_l_dy*mu0: {boundary_l * VACUUM_PERMEABILITY}")
print(f"boundary_h_dy*VACUUM_PERMEABILITY: {boundary_h * VACUUM_PERMEABILITY}")

alfven_impedance = alfven.calculate_alfven_impedance(alfven_velocity=alfven_velocity_dynamic)
alfven_admittance = alfven.calculate_alfven_admittance(alfven_impedance=alfven_impedance)

ionospheric_reflection_coefficient = alfven.calculate_ionospheric_reflection_coefficient(alfven_admittance=alfven_admittance, pedersen_impedance=pedersen_conductance_dynamic)
print(f"reflection_coef:{ionospheric_reflection_coefficient}")

phase_vary_range = alfven.calculate_electric_magnetic_field_phase_difference_range(ionospheric_reflection_coefficient)
print(f"phase_vary_range: {phase_vary_range}")

# %% 2nd plot: define
nrows, ncols_main = 4, 2
plot_defs = [[None for _ in range(ncols_main)] for _ in range(nrows)]
# left is static region and right is dynamic region
# Row 0
plot_defs[0][0] = {
    "plot_type": "line",
    "x_data": datetimes_sta,
    "y_data_list": [
        E_sc1_sta,
        b_sc2_sta,
    ],  # 先用蓝色绘制电场，在用橙色绘制磁场
    "linewidth": 1.8,
    "labels": [r"$E_{x}$", r"$\Delta {B_{y}}$"],
    "title": f"Static Region from\n{st_sta} to {et_sta}",
    "xlabel": "Time (UTC)",
    "ylabel": "Amplitude",
}
plot_defs[0][1] = {
    "plot_type": "line",
    "x_data": datetimes_dy,
    "y_data_list": [E_sc1_dy, b_sc2_dy],
    "labels": [r"$E_{x}$", r"$\Delta {B_{y}}$"],
    "title": f"Dynamic Region from\n{st_dy} to {et_dy}",
    "xlabel": "Time (UTC)",
    "ylabel": "Amplitude",
}
# Row 1
plot_defs[1][0] = {
    "plot_type": "line",
    "x_data": frequencies_psd_sta,
    "y_data_list": [Pxx_E_sc1_sta, Pxx_b_sc2_sta],
    "labels": [r"PSD of $E_{x}$", r"PSD of $\Delta {B_{y}}$"],
    "yscale": "log",  # Use log scale
    "title": "PSD of $\Delta {B_{y}}$ and $E_{x}$",
    "xlabel": "Frequency (Hz)",
    "ylabel": "PSD",
}
plot_defs[1][1] = {
    "plot_type": "line",
    "x_data": frequencies_psd_dy,
    "y_data_list": [Pxx_E_sc1_dy, Pxx_b_sc2_dy],
    "labels": [r"PSD of $E_{x}$", r"PSD of $\Delta {B_{y}}$"],
    "yscale": "log",  # Use log scale
    "title": "PSD of $\Delta {B_{y}}$ and $E_{x}$",
    "xlabel": "Frequency (Hz)",
    "ylabel": "PSD",
}
# Row 2
plot_defs[2][0] = {
    "plot_type": "line",
    "x_data": frequencies_psd_sta,
    "y_data_list": [eb_ratio_psd_sta],
    "yscale": "log",  # Use log scale
    "labels": [r"ratio $\frac{E_{x}}{\Delta {B_{y}}}$"],
    "title": r"ratio $\frac{E_{x}}{\Delta {B_{y}}}$",
    "xlabel": "Frequency (Hz)",
    "ylabel": "ratio",
    "hlines": [
        {"y": boundary_l},
        {
            "y": boundary_h,
        },
        {"y": alfven_velocity_dynamic, "label": r"$v_A$"},
    ],
}
plot_defs[2][1] = {
    "plot_type": "line",
    "x_data": frequencies_psd_dy,
    "y_data_list": [eb_ratio_psd_dy],
    "yscale": "log",
    "labels": [r"ratio $\frac{E_{x}}{\Delta {B_{y}}}$"],
    "title": r"ratio $\frac{E_{x}}{\Delta {B_{y}}}$",
    "xlabel": "Frequency (Hz)",
    "ylabel": "ratio",
    "hlines": [
        {"y": boundary_l},
        {
            "y": boundary_h,
        },
        {"y": alfven_velocity_dynamic, "label": r"$v_A$"},
    ],
}
# Row 3
plot_defs[3][0] = {
    "plot_type": "pcolormesh",
    "x_data": frequencies_spec_sta,
    "y_data": (phase_bins_sta[:-1] + phase_bins_sta[1:]) / 2,
    "z_data": phase_histogram2d_sta.T,
    "title": "The Occurrence of Phase Difference between\n$\Delta {B_{y}}$ and $E_{x}$",
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
    "title": "The Occurrence of Phase Difference between\n$\Delta {B_{y}}$ and $E_{x}$",
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
    nrows=4,  # may modify
    ncols_main=2,
    add_shared_cbar=True,
    shared_cbar_label="Occurrence (Unit 1)",
    figure_title=f"Alfven Wave Case: DMSP {dmsp_number} from {start_time} to {end_time}",
    global_cmap="viridis",
    figsize=(10, 16),
    use_shared_clims=True,  # Use shared clim for spectrograms
    title_fontsize=11,
    label_fontsize=10,
    tick_label_fontsize=9,
    legend_fontsize=9,
    annotation_fontsize=8,
    panel_label_fontsize=11,
    rotate_xticklabels=True
)

# %% 2nd plot: save
if save:
    output_filename_png = f"2nd_plot_Alfven_Wave_Case_DMSP_{dmsp_number}_from_{start_time}_to_{end_time}.png"
    output_path = os.path.join(save_dir, output_filename_png)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saving figure to {output_filename_png} (300 DPI)")

#%% if show
if show:
    plt.show()