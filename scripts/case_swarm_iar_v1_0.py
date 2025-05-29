"""
生成论文所需的结果
"""

# %% import

import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from configs import ProjectConfigs
from utils import spectral

import pyaw.satellite
import pyaw.utils
from src.pyaw import plot_multi_panel, plot_gridded_panels
from utils import histogram2d, coordinate
from utils import (
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
Ehx_outlier = set_outliers_nan_std(Ehx, 1, print_=True)
Ehx_outlier_interp = interpolate_missing(Ehx_outlier, df_e_clip.index.values)

Ehy = df_e_clip["Ehy"].values
Ehy_outlier = set_outliers_nan_std(Ehy, 1, print_=True)
Ehy_outlier_interp = interpolate_missing(Ehy_outlier, df_e_clip.index.values)

VsatN = df_e_clip["VsatN"].values
VsatE = df_e_clip["VsatE"].values

# %%  --- process data: electric field: sc2nec
rotmat_nec2sc, rotmat_sc2nec = pyaw.satellite.NEC2SCandSC2NEC.get_rotmat_nec2sc_sc2nec(
    VsatN, VsatE
)
E_north, E_east = pyaw.satellite.NEC2SCandSC2NEC.do_rotation(
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





# %% preset region parameters for plot annoatations, labels and so on
st_dy = np.datetime64("2016-03-11 06:47:35")
et_dy = np.datetime64("2016-03-11 06:47:55")
st_sta = np.datetime64("2016-03-11 06:47:05")
et_sta = np.datetime64("2016-03-11 06:47:25")

# %% Region: get clip data
t_mask_dy = (datetimes >= st_dy) & (datetimes <= et_dy)
datetimes_dy = datetimes[t_mask_dy]
delta_B_E_align_dy = delta_B_E_align[t_mask_dy]
E_north_dy = E_north[t_mask_dy]

t_mask_sta = (datetimes >= st_sta) & (datetimes <= et_sta)
datetimes_sta = datetimes[t_mask_sta]
delta_B_E_align_sta = delta_B_E_align[t_mask_sta]
E_north_sta = E_north[t_mask_sta]


#%% Region: CWT for IAR
import pywt

fs=16
dt = 1 / fs  # 采样间隔
wavelet = "cmor1.5-1.0"
fc = pywt.central_frequency(wavelet)  # 获取中心频率
# 计算尺度范围
a_min = fc / (8 * dt)  # 对应 8 Hz
a_max = fc / (0.1 * dt)  # 对应 0.1 Hz
scales = np.arange(a_min, a_max, 0.1)  # 生成尺度数组

cwt_eb_dy = spectral.CWT(E_north_dy, delta_B_E_align_dy, scales=scales, fs=fs,wavelet=wavelet)
cwt_eb_m_dy, cwt_eb_p_dy, cwt_eb_f_dy = cwt_eb_dy.get_cross_spectral()

num_bins = 30
cwt_eb_p_bins_dy, cwt_eb_p_histogram2d_dy = pyaw.utils.get_phase_histogram2d(cwt_eb_f_dy, cwt_eb_p_dy, num_bins)

cwt_eb_sta = spectral.CWT(E_north_sta, delta_B_E_align_sta, scales=scales, fs=fs,wavelet=wavelet)
cwt_eb_m_sta, cwt_eb_p_sta, cwt_eb_f_sta = cwt_eb_sta.get_cross_spectral()

cwt_eb_p_bins_sta, cwt_eb_p_histogram2d_sta = pyaw.utils.get_phase_histogram2d(cwt_eb_f_sta, cwt_eb_p_sta, num_bins)


# #%% 1st plot
# fig = plt.figure()
# plt.pcolormesh((cwt_eb_p_bins_dy[:-1] + cwt_eb_p_bins_dy[1:]) / 2, cwt_eb_f_dy, cwt_eb_p_histogram2d_dy, shading='auto',cmap='jet')
# plt.colorbar()
# plt.xlabel('Phase [degree]')
# plt.ylabel('Frequency [Hz]')
# plt.title('Phase Histogram')
#
# #%% 1st plot: save
# save = True
# if save:
#     output_filename_png = f"1st_plot_dy_IAR_Alfven_Wave_Case_SwarmA__from_{start_time}_to_{end_time}.png"
#     output_path = os.path.join(save_dir, output_filename_png)
#     print(f"Saving figure to {output_filename_png} (300 DPI)")
#     fig.savefig(output_path, dpi=300, bbox_inches="tight")

#%% 1st plot: use imshow()
fig = plt.figure(figsize=(6,4))
cwt_eb_p_histogram2d_dy_norm = cwt_eb_p_histogram2d_dy / np.max(cwt_eb_p_histogram2d_dy)
plt.imshow(cwt_eb_p_histogram2d_dy_norm, extent=(-180, 180, cwt_eb_f_dy[-1], cwt_eb_f_dy[0]), aspect='auto', cmap='jet')
plt.title("Phase of CWT")
plt.xlabel("Phase (Degree)")
plt.ylabel("Frequency [Hz]")
plt.colorbar(label="Probability of Occurrence")
save = True
if save:
    output_filename_png = f"1st_plot_imshow_sta_IAR_Alfven_Wave_Case_SwarmA__from_{start_time}_to_{end_time}.png"
    output_path = os.path.join(save_dir, output_filename_png)
    print(f"Saving figure to {output_filename_png} (300 DPI)")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")