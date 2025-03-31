#%%
import os.path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import spectrogram

from configs import ProjectConfigs
from core import zh1
from pyaw.parameters import (
    PhysicalParameters,
    calculate_lower_bound,
    calculate_upper_bound,
    calculate_R,
    calculate_phase_vary_range,
)
from pyaw.utils import spectral
from pyaw.utils.plot import plot_multi_panel, plot_gridded_panels
from utils import histogram2d
#%%

# %% basic parameters
window = "hann"
save_dir = r"G:\note\毕业论文\images"
#%%

data_dir_path = ProjectConfigs.data_dir_path
orbit_num = 175371
file_name_scm = "CSES_01_SCM_1_L02_A2_175371_20210331_234620_20210401_002156_000.h5"
file_name_efd = "CSES_01_EFD_1_L2A_A1_175371_20210331_234716_20210401_002158_000.h5"
file_path_scm = os.path.join(data_dir_path, file_name_scm)
file_path_efd = os.path.join(data_dir_path, file_name_efd)

scm = zh1.SCM(file_path_scm)
efd = zh1.EFD(file_path_efd)
#%%
# from df1c_split_list to choose st,et for SCMEFDUlf
# "2021-03-31 23:47:14.468","2021-03-31 23:51:18.706"
# "2021-03-31 23:47:16.352","2021-03-31 23:50:26.816"
st = pd.Timestamp("2021-03-31 23:47:16.352")
et = pd.Timestamp("2021-03-31 23:50:26.816")
#%%
scm_efd = zh1.SCMEFDUlf(st=st,et=et,fp_scm=file_path_scm,fp_efd=file_path_efd)
#%%
df = scm_efd.preprocess_data()
#%% set base datetimes. no lat, MLT(of AACGM)
datetimes = df.index.values
dt = datetime.fromisoformat(str(datetimes[0]))
start_time = dt.strftime("%Y%m%dT%H%M%S")
dt = datetime.fromisoformat(str(datetimes[-1]))
end_time = dt.strftime("%Y%m%dT%H%M%S")

#%% choose pair
b_e = df['b_enu1'].values
E_n = df['e_enu2'].values
assert not np.isnan(b_e).any(), "have nan values"
assert not np.isnan(E_n).any(), "have nan values"

b_n = df['b_enu2'].values
E_e = df['e_enu1'].values
b_u = df['b_enu3'].values
E_u = df['e_enu3'].values

#%% max and min print
print("max b_n:",np.max(b_n)),print("min b_n:",np.min(b_n))

#%%
plt.style.use(
    "seaborn-v0_8-paper"
)  # Good for papers. can not set because the former plot already set

fig,(ax1,ax2) = plt.subplots(2,1,figsize=(10,8), sharex=True)


#%% sub-figure 1
ax1.plot(datetimes,b_e,label=r"$b_e$",linestyle='-')
ax1.plot(datetimes,b_n,label=r"$b_n$",linestyle='--')
ax1.plot(datetimes,b_u,label=r"$b_u$",linestyle=':')
ax1.grid(True)
ax1.set_ylabel("Magnetic Flux Density (nT)")
ax1.set_title("Magnetic Field")
ax1.legend(loc='best')


ax2.plot(datetimes,E_e,label=r"$E_e$", linestyle='-')
ax2.plot(datetimes,E_n,label=r"$E_n$", linestyle='--')
ax2.plot(datetimes,E_u,label=r"$E_u$", linestyle=':')
ax2.grid(True)
ax2.set_xlabel("Time UTC")
ax2.set_ylabel("Electric Field Strength (mV/m)")
ax2.set_title("Electric Field")
ax2.legend(loc='best')

plt.suptitle(f"ZH-1 Magnetic Field and Electric Field from {start_time} to {end_time}")

save = True
if save:
    output_filename_png = f"ZH-1_Magnetic_Field_and_Electric_Field_from_{start_time}_to_{end_time}.png"
    output_path = os.path.join(save_dir, output_filename_png)
    print(f"Saving figure to {output_filename_png} (300 DPI)")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

plt.show()

#
#
# # %% spectrogram settings
# fs = 32
# spectrogram_window_seconds = 2
# nperseg = int(spectrogram_window_seconds * fs)
#
# # %% get spectrogram
# frequencies, ts, Sxx_b = spectrogram(
#     b_e,
#     fs=fs,
#     window=window,
#     nperseg=nperseg,
#     mode="complex",
# )
# _, _, Sxx_e = spectrogram(E_n, fs=fs, window=window, nperseg=nperseg, mode="complex")
#
# # %% spectrogram: get datetime type ndarray for the plot
# ts_dt64 = datetimes[0] + [np.timedelta64(int(_), "s") for _ in ts]
#
# # %% get the cross spectral
# cpsd = Sxx_e * np.conj(Sxx_b)
#
# # %% get Coherency
# segment_length_sec = 2  # 越大最后得到的数组的长度越小
# try:
#     mid_times_all, avg_complex_coh = spectral.calculate_segmented_complex_coherency(
#         datetimes,
#         b_e,
#         E_n,
#         fs=fs,
#         segment_length_sec=segment_length_sec,
#         nfft_coh=int(
#             fs * segment_length_sec * 0.5
#         ),  # Use segment_length_sec/2 second FFT within segments
#     )
#     # Extract magnitude and phase from the complex result
#     avg_coh_magnitude = np.abs(avg_complex_coh)
#
# except ValueError as e:
#     print(f"Error calculating Coherency: {e}")
# except Exception as e:
#     print(f"An unexpected error occurred: {e}")
#
# # %% preset region parameters for plot annoatations, labels and so on
# st_dy = np.datetime64("2014-01-01T01:49:00")
# et_dy = np.datetime64("2014-01-01T01:51:00")
# st_sta = np.datetime64("2014-01-01T01:54:00")
# et_sta = np.datetime64("2014-01-01T01:56:00")
#
# # %% 1st plot: define
#
# subplot_defs = [
#     {
#         "plot_type": "line",
#         "x_data": datetimes,
#         "y_data": b_e,
#         "label": r"$\Delta B_{y}$",
#         "title": "y Component of Disturb Magnetic Field",
#         "ylabel": "Magnetic Flux Density (nT)",
#         "linewidth": 1.8,
#     },  # Slightly thicker line
#     {
#         "plot_type": "line",
#         "x_data": datetimes,  # Assuming same time base for E-field line plot
#         "y_data": E_n,
#         "label": r"$E_{x}$",
#         "title": "x Component of Electric Field",
#         "ylabel": "Electric Field Strength (mV/m)",
#         "linewidth": 1.8,
#     },
#     {
#         "plot_type": "pcolormesh",
#         "x_data": ts_dt64,
#         "y_data": frequencies,
#         "z_data": np.abs(Sxx_b),
#         "title": "Power Spectral Density of y Component of Disturb Magnetic Field",
#         "ylabel": "Frequency (Hz)",
#         "clabel": r"$S_{bb}(f,t)$",
#         "shading": "gouraud",
#     },
#     {
#         "plot_type": "pcolormesh",
#         "x_data": ts_dt64,
#         "y_data": frequencies,
#         "z_data": np.abs(Sxx_e),
#         "title": "Power Spectral Density of x Component of Electric Field",
#         "ylabel": "Frequency (Hz)",
#         "clabel": r"$S_{EE}(f,t)$",
#         "shading": "gouraud",
#     },
#     {
#         "plot_type": "pcolormesh",
#         "x_data": ts_dt64,
#         "y_data": frequencies,
#         "z_data": np.abs(cpsd),
#         "title": r"The Module of Cross Power Spectral Density of $b_y$ and $E_{x}$",
#         "ylabel": "Frequency (Hz)",
#         "clabel": r"$S_{Eb}(f,t)$",
#         "shading": "gouraud",
#     },
#     {
#         "plot_type": "line",
#         "x_data": mid_times_all,
#         "y_data": avg_coh_magnitude,
#         "label": r"Coherency",
#         "title": "Coherency of $\Delta B_{y}$ and $E_{x}$",
#         "ylabel": "Coherency (Unit 1)",
#         "linewidth": 1.8,
#     },  # the coherency subfigure
# ]
#
# # # %% 1st plot: add labels, annotations ...
# # for subplot_def in subplot_defs[:2]:
# #     subplot_def["blocks"] = [
# #         {
# #             "start": st_sta,
# #             "end": et_sta,
# #             "color": "#004488",
# #             "label": "Static Region",
# #         },
# #         {
# #             "start": st_dy,
# #             "end": et_dy,
# #             "color": "#DDAA33",
# #             "label": "Dynamic Region",
# #         },
# #     ]
# # for subplot_def in subplot_defs[2:-1]:
# #     subplot_def["vlines"] = {
# #         "Static Region Start": st_sta,
# #         "Static Region End": et_sta,
# #         "Dynamic Region Start": st_dy,
# #         "Dynamic Region End": et_dy,
# #     }
# # subplot_defs[-1]["blocks"] = [
# #     {
# #         "start": st_sta,
# #         "end": et_sta,
# #         "color": "#004488",
# #         "label": "Static Region",
# #     },
# #     {
# #         "start": st_dy,
# #         "end": et_dy,
# #         "color": "#DDAA33",
# #         "label": "Dynamic Region",
# #     },
# # ]
# # subplot_defs[-1]["hlines"] = [{"y": 0.5, "color": "magenta", "linestyle": "-."}]
#
# # %% 1st plot: plot settings
# plt.style.use("seaborn-v0_8-paper")  # Good for papers
#
# # %% 1st plot: Call the function to plot
# fig, axes = plot_multi_panel(
#     subplot_definitions=subplot_defs,
#     # Reference times for aux data (note that lats,mlts based on datetimes not
#     # tsdt64)
#     x_datetime_ref=datetimes,
#     x_label_step=100,  # Show label every N points
#     figsize=(10, 14),
#     # figure_title=f"Alfven Wave Case: DMSP {dmsp_number} from {start_time} to {end_time}",
#     global_cmap="viridis",
#     # global_vmin=-20,
#     # global_vmax=20,
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
#
# # %% 1st plot: save fig with high DPI
# save = True
# if save:
#     output_filename_png = "1st_zh1_test"
#     output_path = os.path.join(save_dir, output_filename_png)
#     print(f"Saving figure to {output_filename_png} (300 DPI)")
#     fig.savefig(output_path, dpi=300, bbox_inches="tight")