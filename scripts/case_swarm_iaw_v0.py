import os.path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.signal import spectrogram

from configs import ProjectConfigs
from pyaw.utils import spectral
from utils import histogram2d, coordinate
from utils.other import (
    OutlierData,
    interpolate_missing,
    align_high2low,
    get_3arrs,
)

# basic parameters
WINDOW = "hann"

SAVE_DIR = r"G:\note\毕业论文\images"

SWARM_TYPE = "A"

# plot settings
# Other options: 'ggplot', 'seaborn-v0_8-talk', 'default'
# See available: print(plt.style.available)
plt.style.use("seaborn-v0_8-paper")  # Good for papers

# file_paths
data_dir_path = ProjectConfigs.data_dir_path
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

# read data as df
df_b = pd.read_pickle(file_path_b)
df_b_aux = pd.read_pickle(file_path_b_aux)
df_b_IGRF = pd.read_pickle(file_path_b_igrf)
df_e = pd.read_pickle(file_path_tct16)
df_e_aux = pd.read_pickle(file_path_tct16_aux)

# process data: clip needed data for efficiency
df_b_clip = df_b[["B_NEC", "Longitude", "Latitude", "Radius", "q_NEC_CRF"]]
df_b_aux_clip = df_b_aux[["QDLat", "QDLon", "MLT"]]
df_b_IGRF_clip = df_b_IGRF[["B_NEC_IGRF"]]
df_e_clip = df_e[
    ["Longitude", "Latitude", "Radius", "VsatE", "VsatN", "VsatC", "Ehy", "Ehx"]
]
df_e_aux_clip = df_e_aux[["QDLat", "QDLon", "MLT"]]

# process data: use time to clip data again
start_time = "20160311T064700"
end_time = "20160311T064900"
df_b_clip = df_b_clip.loc[pd.Timestamp(start_time) : pd.Timestamp(end_time)]
df_b_aux_clip = df_b_aux_clip.loc[pd.Timestamp(start_time) : pd.Timestamp(end_time)]
df_b_IGRF_clip = df_b_IGRF_clip.loc[pd.Timestamp(start_time) : pd.Timestamp(end_time)]
df_e_clip = df_e_clip.loc[pd.Timestamp(start_time) : pd.Timestamp(end_time)]
df_e_aux_clip = df_e_aux_clip.loc[pd.Timestamp(start_time) : pd.Timestamp(end_time)]

latitudes = df_e_clip["Latitude"].values
mlts = df_e_aux_clip["MLT"].values

# process data: electric field
Ehx = df_e_clip["Ehx"].values
Ehx_outlier = OutlierData.set_outliers_nan_std(Ehx, 1, print_=True)
Ehx_outlier_interp = interpolate_missing(Ehx_outlier, df_e_clip.index.values)

Ehy = df_e_clip["Ehy"].values
Ehy_outlier = OutlierData.set_outliers_nan_std(Ehy, 1, print_=True)
Ehy_outlier_interp = interpolate_missing(Ehy_outlier, df_e_clip.index.values)

# get velocity of satellite
VsatN = df_e_clip["VsatN"].values
VsatE = df_e_clip["VsatE"].values
VsatC = df_e_clip[
    "VsatC"
].values  # 因为base datetimes是datetimes_e，所以着3个速度分量不用进行时间对齐

# process data: electric field: sc2nec
rotmat_nec2sc, rotmat_sc2nec = coordinate.NEC2SCandSC2NEC.get_rotmat_nec2sc_sc2nec(
    VsatN, VsatE
)
E_north, E_east = coordinate.NEC2SCandSC2NEC.do_rotation(
    -Ehx_outlier_interp, -Ehy_outlier_interp, rotmat_sc2nec
)  # todo: why need '-'

# process data: magnetic field
B_N, B_E, B_C = get_3arrs(df_b_clip["B_NEC"].values)
B_N_IGRF, B_E_IGRF, B_C_IGRF = get_3arrs(df_b_IGRF_clip["B_NEC_IGRF"].values)
delta_B_E = B_E - B_E_IGRF
delta_B_N = B_N - B_N_IGRF

# magnetic field: downsample, use align time method
datetimes_e = df_e_clip.index.values
datetimes_b = df_b_clip.index.values
delta_B_E_align = align_high2low(delta_B_E, datetimes_b, datetimes_e)
delta_B_N_align = align_high2low(delta_B_N, datetimes_b, datetimes_e)
# also get back B (all components for strength and potential va calculate)
B_E_IGRF_align = align_high2low(B_E_IGRF, datetimes_b, datetimes_e)
B_N_IGRF_align = align_high2low(B_N_IGRF, datetimes_b, datetimes_e)
B_C_IGRF_align = align_high2low(B_C_IGRF, datetimes_b, datetimes_e)

# base datetimes
datetimes = datetimes_e

# spectrogram: settings
FS = 16
SPECTROGRAM_WINDOW_SECONDS = 4
NPERSEG = int(SPECTROGRAM_WINDOW_SECONDS * FS)

# choose a disturb magnetic field and electric field pair and get spectrogram
frequencies, ts, Sxx_b = spectrogram(
    delta_B_E_align,
    fs=FS,
    window=WINDOW,
    nperseg=NPERSEG,
    mode="complex",
)
_, _, Sxx_e = spectrogram(
    E_north, fs=FS, window=WINDOW, nperseg=NPERSEG, mode="complex"
)

# get datetime type ndarray for the plot
ts_dt64 = datetimes[0] + [np.timedelta64(int(_), "s") for _ in ts]

# get the cross spectral density
cpsd = Sxx_e * np.conj(Sxx_b)

# use new method to get Coherency
segment_length_sec = 4  # 越大最后得到的数组的长度越小，取和之前的spectrogram输入的窗口长度是一个不错的选择
try:
    mid_times_all, avg_complex_coh = spectral.calculate_segmented_complex_coherency(
        datetimes,
        delta_B_E_align,
        E_north,
        fs=FS,
        segment_length_sec=segment_length_sec,
        nfft_coh=int(
            FS * segment_length_sec * 0.5
        ),  # Use segment_length_sec/2 second FFT within segments
    )
    # Extract magnitude and phase from the complex result
    avg_coh_magnitude = np.abs(avg_complex_coh)

except ValueError as e:
    print(f"Error calculating Coherency: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# preset region parameters for plot annoatations, labels and so on
ST_DY = np.datetime64("2016-03-11 06:47:35")
ET_DY = np.datetime64("2016-03-11 06:47:55")
ST_STA = np.datetime64("2016-03-11 06:47:05")
ET_STA = np.datetime64("2016-03-11 06:47:25")

# 1st plot: define
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

# add labels, annotations ...
for subplot_def in subplot_defs[:2]:
    subplot_def["blocks"] = [
        {
            "start": ST_STA,
            "end": ET_STA,
            "color": "#004488",
            "label": "Static Region",
        },
        {
            "start": ST_DY,
            "end": ET_DY,
            "color": "#DDAA33",
            "label": "Dynamic Region",
        },
    ]
for subplot_def in subplot_defs[2:-1]:
    subplot_def["vlines"] = {
        "Static Region Start": ST_STA,
        "Static Region End": ET_STA,
        "Dynamic Region Start": ST_DY,
        "Dynamic Region End": ET_DY,
    }
subplot_defs[-1]["blocks"] = [
    {
        "start": ST_STA,
        "end": ET_STA,
        "color": "#004488",
        "label": "Static Region",
    },
    {
        "start": ST_DY,
        "end": ET_DY,
        "color": "#DDAA33",
        "label": "Dynamic Region",
    },
]
subplot_defs[-1]["hlines"] = [{"y": 0.5, "color": "magenta", "linestyle": "-."}]

# Define Aux Data for X labels
aux_data_for_x = {"Lat": latitudes, "MLT": mlts}


# %% 1st plot: Call the function to plot
# fig, axes = plot_multi_panel(
#     subplot_definitions=subplot_defs,
#     x_datetime_ref=datetimes,  # Reference times for aux data (note that lats,mlts based on datetimes not tsdt64)
#     x_aux_data=aux_data_for_x,  # The aux data arrays
#     x_label_step=200,  # Show label every N points
#     figsize=(10, 14),
#     figure_title=f"Alfven Wave Case: Swarm{swarm_type} from {start_time} to {end_time}",
#     # global_cmap=matplotlib.rcParams["image.cmap"],  # default
#     global_cmap="viridis",
#     global_vmin=-10,  # Example: Manually set global limits
#     global_vmax=10,
#     use_shared_clims=True,  # Use calculated shared limits unless overridden
#     rotate_xticklabels=0,  # Keep horizontal for concise formatter
#     # font size
#     title_fontsize=11,
#     label_fontsize=10,
#     tick_label_fontsize=9,
#     legend_fontsize=9,
#     annotation_fontsize=8,
#     panel_label_fontsize=11,  # Control (a), (b)... size
# )

# # %% 1st plot: save fig with high DPI
# save = False
# if save:
#     output_filename_png = f"1st_plot_Alfven_Wave_Case_Swarm{swarm_type}_from_{start_time}_to_{end_time}.png"
#     output_path = os.path.join(save_dir, output_filename_png)
#     print(f"Saving figure to {output_filename_png} (300 DPI)")
#     fig.savefig(output_path, dpi=300, bbox_inches="tight")

# Region: get clip data
t_mask_dy = (datetimes >= ST_DY) & (datetimes <= ET_DY)
datetimes_dy = datetimes[t_mask_dy]
delta_B_E_align_dy = delta_B_E_align[t_mask_dy]
E_north_dy = E_north[t_mask_dy]

t_mask_sta = (datetimes >= ST_STA) & (datetimes <= ET_STA)
datetimes_sta = datetimes[t_mask_sta]
delta_B_E_align_sta = delta_B_E_align[t_mask_sta]
E_north_sta = E_north[t_mask_sta]
# also get back B (just for dynamic region) (all components for strength and potential va calculate)
B_E_IGRF_align_dy = B_E_IGRF_align[t_mask_dy]
B_N_IGRF_align_dy = B_N_IGRF_align[t_mask_dy]
B_C_IGRF_align_dy = B_C_IGRF_align[t_mask_dy]
B_strength_align_dy = np.sqrt(
    B_E_IGRF_align_dy**2 + B_N_IGRF_align_dy**2 + B_C_IGRF_align_dy**2
)
B_strength_align_dy_value = np.mean(B_strength_align_dy)

# velocity clip (just for dynamic region)
VsatN_dy = VsatN[t_mask_dy]
VsatE_dy = VsatE[t_mask_dy]
VsatC_dy = VsatC[t_mask_dy]
Vsat = np.sqrt(VsatN_dy**2 + VsatE_dy**2 + VsatC_dy**2)
Vsat_mean = np.mean(Vsat)


# get psd
nperseg_psd = 64  # same as the 1st spectrogram nperseg

delta_B_E_align_dy_psd = spectral.PSD(
    delta_B_E_align_dy,
    fs=FS,
    nperseg=nperseg_psd,
    window=WINDOW,
    scaling="density",
)  # same arguments setting as spectrogram
E_north_dy_psd = spectral.PSD(
    E_north_dy,
    fs=FS,
    nperseg=nperseg_psd,
    window=WINDOW,
    scaling="density",
)
delta_B_E_align_sta_psd = spectral.PSD(
    delta_B_E_align_sta,
    fs=FS,
    nperseg=nperseg_psd,
    window=WINDOW,
    scaling="density",
)
E_north_sta_psd = spectral.PSD(
    E_north_sta,
    fs=FS,
    nperseg=nperseg_psd,
    window=WINDOW,
    scaling="density",
)

frequencies_psd_dy, Pxx_delta_B_E_align_dy = delta_B_E_align_dy_psd.get_psd()
_, Pxx_E_north_dy = E_north_dy_psd.get_psd()
frequencies_psd_sta, Pxx_delta_B_E_align_sta = delta_B_E_align_sta_psd.get_psd()
_, Pxx_E_north_sta = E_north_sta_psd.get_psd()

# cpsd
SPECTROGRAM_WINDOW_SECONDS = 4  # compare to the former spectrogram 4
NPERSEG = int(SPECTROGRAM_WINDOW_SECONDS * FS)

frequencies_spec_dy, ts_dy, Sxx_b_dy = spectrogram(
    delta_B_E_align_dy,
    fs=FS,
    window=WINDOW,
    nperseg=NPERSEG,
    mode="complex",
)

_, _, Sxx_e_dy = spectrogram(
    E_north_dy, fs=FS, window=WINDOW, nperseg=NPERSEG, mode="complex"
)

frequencies_spec_sta, ts_sta, Sxx_b_sta = spectrogram(
    delta_B_E_align_sta,
    fs=FS,
    window=WINDOW,
    nperseg=NPERSEG,
    mode="complex",
)
_, _, Sxx_e_sta = spectrogram(
    E_north_sta,
    fs=FS,
    window=WINDOW,
    nperseg=NPERSEG,
    mode="complex",
)

ts_dt64_dy = datetimes_dy[0] + [np.timedelta64(int(_), "s") for _ in ts_dy]
ts_dt64_sta = datetimes_sta[0] + [np.timedelta64(int(_), "s") for _ in ts_sta]

cpsd_dy = Sxx_e_dy * np.conj(Sxx_b_dy)
cpsd_sta = Sxx_e_sta * np.conj(Sxx_b_sta)

# phase difference between b and E
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

# ratio

eb_ratio_psd_dy = (
    (Pxx_E_north_dy / Pxx_delta_B_E_align_dy) * 1e-3 * 1e9
)  # transform unit
eb_ratio_psd_sta = (Pxx_E_north_sta / Pxx_delta_B_E_align_sta) * 1e-3 * 1e9

# lower and upper bound and other parameters
from pyaw.parameters import VACUUM_PERMEABILITY, Alfven

pedersen_conductance_dynamic = 3.0
# pedersen_conductance_static = 0.5
va_dy = 1.4e6
# va_sta = 1.3e6

alfven = Alfven()
(
    general_dynamic_lower_boundary,
    general_dynamic_upper_boundary,
    general_static_lower_boundary,
    general_static_upper_boundary,
) = alfven.general_dynamic_static_boundary()
print(f"boundary_l_dy*mu0: {general_dynamic_lower_boundary * VACUUM_PERMEABILITY}")
print(f"boundary_h_dy*mu0: {general_dynamic_upper_boundary * VACUUM_PERMEABILITY}")


alfven_impedance = alfven.calculate_alfven_impedance(alfven_velocity=va_dy)
alfven_admittance = alfven.calculate_alfven_admittance(
    alfven_impedance=alfven_impedance
)
ionospheric_reflection_coefficient = (
    alfven.calculate_ionospheric_reflection_coefficient(
        alfven_admittance=alfven_admittance,
        pedersen_impedance=pedersen_conductance_dynamic,
    )
)

phase_vary_range_dy = alfven.calculate_electric_magnetic_field_phase_difference_range(
    ionospheric_reflection_coefficient=ionospheric_reflection_coefficient
)
# phase_vary_range_sta = calculate_phase_vary_range(reflection_coef_sta)
print(f"phase_vary_range_dy: {phase_vary_range_dy}")
# print(f"phase_vary_range_sta: {phase_vary_range_sta}")

# get iaw curve


from pyaw.parameters import (
    calculate_approx_perpendicular_wavenumber,
    calculate_inertial_alfven_wave_electric_magnetic_field_ratio,
    calculate_electron_inertial_length,
    calculate_electron_plasma_frequency,
    LIGHT_SPEED,
    calculate_ion_thermal_gyroradius,
    OXYGEN_ATOMIC_MASS,
    calculate_ion_gyrofrequency,
)

# electron_number_density = 1e5
electron_number_density = 3e10
# electron_number_density = 1e11
# electron_number_density = 1e12

perpendicular_wavenumber = calculate_approx_perpendicular_wavenumber(
    wave_frequency=frequencies_psd_dy, spacecraft_speed=Vsat_mean
)

electron_plasma_frequency = calculate_electron_plasma_frequency(
    electron_number_density=electron_number_density
)

electron_inertial_length = calculate_electron_inertial_length(
    electron_plasma_frequency=electron_plasma_frequency, light_speed=LIGHT_SPEED
)

inertial_alfven_wave_electric_magnetic_field_ratio = (
    calculate_inertial_alfven_wave_electric_magnetic_field_ratio(
        alfven_velocity=va_dy,
        perpendicular_wavenumber=perpendicular_wavenumber,
        electron_inertial_length=electron_inertial_length,
    )
)

ion_temperature = 1000
ion_mass = OXYGEN_ATOMIC_MASS

ion_gyrofrequency = calculate_ion_gyrofrequency(
    background_magnetic_field=B_strength_align_dy_value * 1e-9, ion_mass=ion_mass
)

ion_thermal_gyroradius = calculate_ion_thermal_gyroradius(
    ion_temperature=ion_temperature,
    ion_mass=ion_mass,
    ion_gyrofrequency=ion_gyrofrequency,
)

inertial_alfven_wave_with_larmor_electric_magnetic_field_ratio = (
    calculate_inertial_alfven_wave_electric_magnetic_field_ratio(
        alfven_velocity=va_dy,
        perpendicular_wavenumber=perpendicular_wavenumber,
        electron_inertial_length=electron_inertial_length,
        ion_thermal_gyroradius=ion_thermal_gyroradius,
    )
)


# plot
plt.figure(figsize=(6, 4))
plt.plot(
    frequencies_psd_dy,
    eb_ratio_psd_dy,
    label=r"ratio $\frac{E_{North}}{\Delta {B_{East}}}$",
)
plt.plot(
    frequencies_psd_dy,
    inertial_alfven_wave_with_larmor_electric_magnetic_field_ratio,
    label="IAW Curve",
)
plt.axhline(y=general_dynamic_lower_boundary, color="magenta", linestyle="-.")
plt.axhline(y=general_dynamic_upper_boundary, color="magenta", linestyle="-.")
plt.axhline(y=va_dy, label=r"$v_A$", color="magenta", linestyle="-.")
plt.yscale("log")
plt.grid(linestyle=":", alpha=0.6)
plt.title(r"ratio $\frac{E_{North}}{\Delta {B_{East}}}$ and IAW Curve")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Ratio")
plt.legend()

# save and show
save = False
if save:
    output_filename_png = f"Inertial_Alfven_Wave_Case_Swarm{SWARM_TYPE}_from_{start_time}_to_{end_time}.png"
    output_path = os.path.join(SAVE_DIR, output_filename_png)
    print(f"Saving figure to {output_filename_png} (300 DPI)")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

plt.show()
