import os

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

import pyaw.satellite

SHOW = False
SAVE = True
ADD_MEASUREMENT_RATIO = True
SAVE_DIR = Path(r"G:\google_drive\mydrive\my_thesis\figures_\plasma_parameters")

# dynamic region time range
ST_DY = np.datetime64("2016-03-11 06:47:35")
ET_DY = np.datetime64("2016-03-11 06:47:55")

script_dir = Path(__file__).parent
data_dir = script_dir / ".." / ".." / "data"

efia_lp_1b_file_name = "SW_OPER_EFIA_LP_1B_12885_20160311T061733_20160311T075106.pkl"
efiatie_2_file_name = "SW_OPER_EFIATIE_2__12885_20160311T061733_20160311T075106.pkl"

efia_lp_1b_file_path = data_dir / efia_lp_1b_file_name
efiatie_2_file_path = data_dir / efiatie_2_file_name

df_lp_1b = pd.read_pickle(efia_lp_1b_file_path)
df_tie_2 = pd.read_pickle(efiatie_2_file_path)

# clip df
df_lp_1b = df_lp_1b[(df_lp_1b.index >= ST_DY) & (df_lp_1b.index <= ET_DY)]
df_tie_2 = df_tie_2[(df_tie_2.index >= ST_DY) & (df_tie_2.index <= ET_DY)]


plt.style.use("seaborn-v0_8-paper")

# plot some parameters
fig, axes = plt.subplots(
    nrows=3, ncols=1, figsize=(12, 10), sharex=True, gridspec_kw={"hspace": 0.15}
)

# subplot1
axes[0].plot(df_lp_1b.index, df_lp_1b["Ne"] * 1e6, color="#1f77b4", linewidth=2)
axes[0].set_ylabel(r"$n_e$ ($m^{-3}$)", fontsize=12)
axes[0].set_yscale("log")
axes[0].grid(True, which="both", ls="--", alpha=0.3)
axes[0].tick_params(axis="both", labelsize=10)
mean_ne = df_lp_1b["Ne"].mean() * 1e6
axes[0].text(
    0.80,
    0.90,
    f"Mean: {mean_ne:.2e}",
    transform=axes[0].transAxes,
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
)

# subplot2
axes[1].plot(df_tie_2.index, df_tie_2["Te_adj_LP"], color="#ff7f0e", linewidth=2)
axes[1].set_ylabel(r"$T_e$ (K)", fontsize=12)
axes[1].grid(True, which="both", ls="--", alpha=0.3)
axes[1].tick_params(axis="both", labelsize=10)
mean_te = df_tie_2["Te_adj_LP"].mean()
axes[1].text(
    0.80,
    0.90,
    f"Mean: {mean_te:.0f}",
    transform=axes[1].transAxes,
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
)

# subplot3
axes[2].plot(df_tie_2.index, df_tie_2["Ti_meas_drift"], color="#2ca02c", linewidth=2)
axes[2].set_ylabel(r"$T_i$ (K)", fontsize=12)
axes[2].grid(True, which="both", ls="--", alpha=0.3)
axes[2].tick_params(axis="both", labelsize=10)
mean_ti = df_tie_2["Ti_meas_drift"].mean()
axes[2].text(
    0.80,
    0.90,
    f"Mean: {mean_ti:.0f}",
    transform=axes[2].transAxes,
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
)

axes[-1].set_xlabel("Time (UTC)", fontsize=12)

time_range = f"{ST_DY} - {ET_DY}"
fig.text(0.99, 0.01, time_range, ha="right", va="bottom", fontsize=8, alpha=0.7)

if SAVE:
    # figure1
    output_filename_png = f"plasma_parameters.png"
    output_path = SAVE_DIR / output_filename_png
    print(f"Saving figure to {output_filename_png} (300 DPI)")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

from src.pyaw import (
    OXYGEN_ATOMIC_MASS,
    calculate_electron_inertial_length,
    LIGHT_SPEED,
    calculate_electron_plasma_frequency,
    calculate_ion_thermal_gyroradius,
    calculate_ion_gyrofrequency,
    calculate_ion_acoustic_gyroradius,
    Alfven,
    calculate_plasma_density,
    calculate_approx_perpendicular_wavenumber,
    calculate_inertial_alfven_wave_electric_magnetic_field_ratio,
    calculate_kinetic_alfven_wave_electric_magnetic_field_ratio,
)

# some parameters set before calculate
electron_number_density = mean_ne
electron_temperature = mean_te
ion_temperature = mean_ti

ion_mass = OXYGEN_ATOMIC_MASS

background_magnetic_field = 4.73e-5

# calculate
electron_plasma_frequency = calculate_electron_plasma_frequency(
    electron_number_density=electron_number_density
)

electron_inertial_length = calculate_electron_inertial_length(
    electron_plasma_frequency=electron_plasma_frequency, light_speed=LIGHT_SPEED
)

ion_gyrofrequency = calculate_ion_gyrofrequency(
    background_magnetic_field=background_magnetic_field, ion_mass=ion_mass
)

ion_thermal_gyroradius = calculate_ion_thermal_gyroradius(
    ion_temperature=ion_temperature,
    ion_mass=ion_mass,
    ion_gyrofrequency=ion_gyrofrequency,
)

ion_acoustic_gyroradius = calculate_ion_acoustic_gyroradius(
    electron_temperature=electron_temperature,
    ion_mass=ion_mass,
    ion_gyrofrequency=ion_gyrofrequency,
)
print(f"electron_plasma_frequency: {electron_plasma_frequency:.2f} rad/s")
print(f"electron_inertial_length: {electron_inertial_length:.2f} m")
print(f"ion_gyrofrequency: {ion_gyrofrequency:.2f} rad/s")
print(f"ion_thermal_gyroradius: {ion_thermal_gyroradius:.2f} m")
print(f"ion_acoustic_gyroradius: {ion_acoustic_gyroradius:.2f} m")

if not ADD_MEASUREMENT_RATIO:
    wave_frequencies = np.arange(0, 8.25, 0.25)
else:
    file_path_b = os.path.join(
        data_dir, "SW_OPER_MAGA_HR_1B_12885_20160311T061733_20160311T075106.pkl"
    )
    file_path_b_igrf = os.path.join(
        data_dir, "IGRF_SW_OPER_MAGA_HR_1B_12885_20160311T061733_20160311T075106.pkl"
    )
    file_path_tct16 = os.path.join(
        data_dir, "SW_EXPT_EFIA_TCT16_12885_20160311T061733_20160311T075106.pkl"
    )
    df_b = pd.read_pickle(file_path_b)
    df_b_IGRF = pd.read_pickle(file_path_b_igrf)
    df_e = pd.read_pickle(file_path_tct16)
    df_b_clip = df_b[["B_NEC", "Longitude", "Latitude", "Radius", "q_NEC_CRF"]]
    df_b_IGRF_clip = df_b_IGRF[["B_NEC_IGRF"]]
    df_e_clip = df_e[
        ["Longitude", "Latitude", "Radius", "VsatE", "VsatN", "VsatC", "Ehy", "Ehx"]
    ]
    start_time = "20160311T064700"
    end_time = "20160311T064900"
    df_b_clip = df_b_clip.loc[pd.Timestamp(start_time) : pd.Timestamp(end_time)]
    df_b_IGRF_clip = df_b_IGRF_clip.loc[
        pd.Timestamp(start_time) : pd.Timestamp(end_time)
    ]
    df_e_clip = df_e_clip.loc[pd.Timestamp(start_time) : pd.Timestamp(end_time)]

    from src.pyaw import OutlierData, interpolate_missing

    Ehx = df_e_clip["Ehx"].values
    Ehx_outlier = set_outliers_nan_std(Ehx, 1, print_=True)
    Ehx_outlier_interp = interpolate_missing(Ehx_outlier, df_e_clip.index.values)
    Ehy = df_e_clip["Ehy"].values
    Ehy_outlier = set_outliers_nan_std(Ehy, 1, print_=True)
    Ehy_outlier_interp = interpolate_missing(Ehy_outlier, df_e_clip.index.values)

    VsatN = df_e_clip["VsatN"].values
    VsatE = df_e_clip["VsatE"].values
    VsatC = df_e_clip["VsatC"].values

    from utils import coordinate

    rotmat_nec2sc, rotmat_sc2nec = pyaw.satellite.NEC2SCandSC2NEC.get_rotmat_nec2sc_sc2nec(
        VsatN, VsatE
    )
    E_north, E_east = pyaw.satellite.NEC2SCandSC2NEC.do_rotation(
        -Ehx_outlier_interp, -Ehy_outlier_interp, rotmat_sc2nec
    )

    from src.pyaw import get_3arrs, align_high2low

    B_N, B_E, B_C = get_3arrs(df_b_clip["B_NEC"].values)
    B_N_IGRF, B_E_IGRF, B_C_IGRF = get_3arrs(df_b_IGRF_clip["B_NEC_IGRF"].values)
    delta_B_E = B_E - B_E_IGRF

    datetimes_e = df_e_clip.index.values
    datetimes_b = df_b_clip.index.values
    delta_B_E_align = align_high2low(delta_B_E, datetimes_b, datetimes_e)
    B_E_IGRF_align = align_high2low(B_E_IGRF, datetimes_b, datetimes_e)
    datetimes = datetimes_e
    FS = 16
    nperseg_psd = 64

    t_mask_dy = (datetimes >= ST_DY) & (datetimes <= ET_DY)
    datetimes_dy = datetimes[t_mask_dy]
    delta_B_E_align_dy = delta_B_E_align[t_mask_dy]
    E_north_dy = E_north[t_mask_dy]

    from utils import spectral

    WINDOW = "hann"
    delta_B_E_align_dy_psd = spectral.PSD(
        delta_B_E_align_dy,
        fs=FS,
        nperseg=nperseg_psd,
        window=WINDOW,
        scaling="density",
    )
    E_north_dy_psd = spectral.PSD(
        E_north_dy,
        fs=FS,
        nperseg=nperseg_psd,
        window=WINDOW,
        scaling="density",
    )
    frequencies_psd_dy, Pxx_delta_B_E_align_dy = delta_B_E_align_dy_psd.get_psd()
    _, Pxx_E_north_dy = E_north_dy_psd.get_psd()
    eb_ratio_psd_dy = (Pxx_E_north_dy / Pxx_delta_B_E_align_dy) * 1e-3 * 1e9
    wave_frequencies = frequencies_psd_dy

satellite_velocity = 7.6e3

perpendicular_wavenumber = calculate_approx_perpendicular_wavenumber(
    wave_frequency=wave_frequencies, spacecraft_speed=satellite_velocity
)


alfven = Alfven()
plasma_density = calculate_plasma_density(
    hydrogen_ion_number_density=0,
    helium_ion_number_density=0,
    oxygen_ion_number_density=electron_number_density,
)
alfven_velocity = alfven.calculate_alfven_velocity(
    background_magnetic_field=background_magnetic_field, plasma_density=plasma_density
)
alfven_impedance = alfven.calculate_alfven_impedance(alfven_velocity=alfven_velocity)
alfven_admittance = alfven.calculate_alfven_admittance(
    alfven_impedance=alfven_impedance
)

pedersen_conductance = 3
ionospheric_reflection_coefficient = (
    alfven.calculate_ionospheric_reflection_coefficient(
        alfven_admittance=alfven_admittance, pedersen_impedance=pedersen_conductance
    )
)

lower_boundary = alfven.calculate_lower_boundary(
    pedersen_conductance=pedersen_conductance
)
upper_boundary = alfven.calculate_upper_boundary(
    alfven_velocity=alfven_velocity, pedersen_conductance=pedersen_conductance
)

inertial_alfven_wave_electric_magnetic_field_ratio = (
    calculate_inertial_alfven_wave_electric_magnetic_field_ratio(
        alfven_velocity=alfven_velocity,
        perpendicular_wavenumber=perpendicular_wavenumber,
        electron_inertial_length=electron_inertial_length,
    )
)

kinetic_alfven_wave_electric_magnetic_field_ratio = (
    calculate_kinetic_alfven_wave_electric_magnetic_field_ratio(
        alfven_velocity=alfven_velocity,
        perpendicular_wavenumber=perpendicular_wavenumber,
        ion_acoustic_gyroradius=ion_acoustic_gyroradius,
    )
)

inertial_alfven_wave_with_larmor_electric_magnetic_field_ratio = (
    calculate_inertial_alfven_wave_electric_magnetic_field_ratio(
        alfven_velocity=alfven_velocity,
        perpendicular_wavenumber=perpendicular_wavenumber,
        electron_inertial_length=electron_inertial_length,
        ion_thermal_gyroradius=ion_thermal_gyroradius,
    )
)

kinetic_alfven_wave_with_larmor_electric_magnetic_field_ratio = (
    calculate_kinetic_alfven_wave_electric_magnetic_field_ratio(
        alfven_velocity=alfven_velocity,
        perpendicular_wavenumber=perpendicular_wavenumber,
        ion_acoustic_gyroradius=ion_acoustic_gyroradius,
        ion_thermal_gyroradius=ion_thermal_gyroradius,
    )
)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
if ADD_MEASUREMENT_RATIO:
    plt.plot(
        wave_frequencies,
        eb_ratio_psd_dy,
        label=r"ratio $\frac{E_{North}}{\Delta {B_{East}}}$",
    )
plt.plot(
    wave_frequencies,
    inertial_alfven_wave_electric_magnetic_field_ratio,
    label="IAW",
    color="blue",
    linestyle="-",
)
plt.plot(
    wave_frequencies,
    inertial_alfven_wave_with_larmor_electric_magnetic_field_ratio,
    label="IAW+Larmor",
    color="red",
    linestyle="--",
)
plt.plot(
    wave_frequencies,
    kinetic_alfven_wave_electric_magnetic_field_ratio,
    label="KAW",
    color="green",
    linestyle=":",
)
plt.plot(
    wave_frequencies,
    kinetic_alfven_wave_with_larmor_electric_magnetic_field_ratio,
    label="KAW+Larmor",
    color="purple",
    linestyle="-.",
)

plt.axhline(alfven_velocity, label="Alfvén velocity", color="orange", linestyle="-")
plt.axhline(lower_boundary, label="Lower boundary", color="gray", linestyle="--")
plt.axhline(upper_boundary, label="Upper boundary", color="black", linestyle=":")
plt.yscale("log")
plt.grid(True, which="both", ls="--", alpha=0.3)
plt.xlabel("Wave Frequency (Hz)", fontsize=12)
plt.ylabel(r"$\frac{E_{\perp}}{\delta B_{\perp}}$ (m/s)", fontsize=12)
if ADD_MEASUREMENT_RATIO:
    plt.title(
        "Wave Mode Electric to Magnetic Field Ratios Add Measurements", fontsize=14
    )
else:
    plt.title("Wave Mode Electric to Magnetic Field Ratios", fontsize=14)
plt.legend(fontsize=10)
plt.tick_params(axis="both", labelsize=10)

if SAVE:
    # figure2
    if not ADD_MEASUREMENT_RATIO:
        output_filename_png = f"wave_mode_electric_to_magnetic_field_ratio.png"
    else:
        output_filename_png = (
            f"wave_mode_electric_to_magnetic_field_ratio_add_measurements_ratio.png"
        )
    output_path = SAVE_DIR / output_filename_png
    print(f"Saving figure to {output_filename_png} (300 DPI)")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")


if SHOW:
    plt.show()
