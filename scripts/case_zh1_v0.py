#%% import
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

# %% basic parameters
window = "hann"
save_dir = r"G:\note\毕业论文\images"

#%% file paths
data_dir_path = ProjectConfigs.data_dir_path
file_name_scm = "CSES_01_SCM_1_L02_A2_175371_20210331_234620_20210401_002156_000.h5"
file_name_efd = "CSES_01_EFD_1_L2A_A1_175371_20210331_234716_20210401_002158_000.h5"
file_path_scm = os.path.join(data_dir_path, file_name_scm)
file_path_efd = os.path.join(data_dir_path, file_name_efd)

#%% read data as df
scm = zh1.SCM(file_path_scm)
efd = zh1.EFD(file_path_efd)

#%% get select data
df1c_list_scm = scm.df1c_split_list
df1c_list_efd = efd.df1c_split_list

datetimes_list_scm = scm.datetimes_split_list
datetimes_list_efd = efd.datetimes_split_list

A231_W_df_split_list,A232_W_df_split_list,A233_W_df_split_list = scm.get_wave_data_split_list()
A111_W_df_split_list,A112_W_df_split_list,A113_W_df_split_list = efd.get_wave_data_split_list()



#%% customize
#%% customize: 1st element of df split list. generate datetimes_e, datetimes_b then align time to a base datetimes.
idx = 0
datetimes_scm = datetimes_list_scm[idx].values
datetimes_efd = datetimes_list_efd[idx].values

Bx = A231_W_df_split_list[idx]  # dataframe
By = A232_W_df_split_list[idx]
Bz = A233_W_df_split_list[idx]
Ex = A111_W_df_split_list[idx]
Ey = A112_W_df_split_list[idx]
Ez = A113_W_df_split_list[idx]

#%% process data: get common time range
start_time_scm = datetimes_scm[0]
end_time_scm = datetimes_scm[-1]

start_time_efd = datetimes_efd[0]
end_time_efd = datetimes_efd[-1]

assert start_time_scm < end_time_efd
assert start_time_efd < end_time_scm

start_time = max(start_time_scm, start_time_efd)
end_time = min(end_time_scm, end_time_efd)

#%% process data: clip
Bx_clip = Bx.loc[start_time:end_time]
By_clip = By.loc[start_time:end_time]
Bz_clip = Bz.loc[start_time:end_time]
Ex_clip = Ex.loc[start_time:end_time]
Ey_clip = Ey.loc[start_time:end_time]
Ez_clip = Ez.loc[start_time:end_time]

#%% generate time index for flatten b,e
interval_b = pd.Timedelta((scm.row_len + 1) / scm.fs, unit='s')
interval_e = pd.Timedelta((efd.row_len + 1) / efd.fs, unit='s')
for i in range(len(Bx_clip.index.values)):
    _ = pd.date_range(start=Bx_clip.index.values[i],periods=4096-1,freq=f"{1 / scm.fs}s")



# #%% get clip datetimes: 1st preview the former scm,efd to choose use which elements in list. then use lat range to get clip df
# mask = scm.df1c_split_list[0]['GEO_LAT'] > -60

#%% get select data
datetimes = scm.datetimes_split_list[2]

#%% get select magnetic and electric field
