import os.path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from utils import other

import pyaw.utils
from src.pyaw import ProjectConfigs

# get data

data_dir_path = ProjectConfigs.data_dir_path
file_name_aux = "aux_SW_OPER_MAGA_LR_1B_12728_20160301T012924_20160301T030258.pkl"
file_name_mea = "SW_OPER_MAGA_LR_1B_12728_20160301T012924_20160301T030258.pkl"
file_name_igrf = "IGRF_SW_OPER_MAGA_LR_1B_12728_20160301T012924_20160301T030258.pkl"
file_path_aux = os.path.join(data_dir_path, file_name_aux)
file_path_mea = os.path.join(data_dir_path, file_name_mea)
file_path_igrf = os.path.join(data_dir_path, file_name_igrf)
data_aux = pd.read_pickle(file_path_aux)
data_mea = pd.read_pickle(file_path_mea)
data_igrf = pd.read_pickle(file_path_igrf)

mlat_data = data_aux["QDLat"].values  # 您的磁纬度数据
mlon_data = data_aux["QDLon"].values  # 您的磁经度数据
mea_B_data_north, mea_B_data_east, _ = pyaw.utils.get_3arrays(data_mea["B_NEC"].values)
igrf_B_data_north, igrf_B_data_east, _ = pyaw.utils.get_3arrays(
    data_igrf["B_NEC_IGRF"].values
)
delta_B_data_north, delta_B_data_east = (
    mea_B_data_north - igrf_B_data_north,
    mea_B_data_east - igrf_B_data_east,
)

delta_B_data = delta_B_data_north  # 您的扰动磁场数据
# delta_B_data = delta_B_data_east

# 准备网格
mlat_grid = np.linspace(50, 90, 100)
mlon_grid = np.linspace(0, 360, 100)
MLAT, MLON = np.meshgrid(mlat_grid, mlon_grid)

# 插值
delta_B_grid = griddata(
    (mlat_data, mlon_data), delta_B_data, (MLAT, MLON), method="linear"
)

# 绘图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
ax.set_extent([-180, 180, 50, 90], ccrs.PlateCarree())
ax.coastlines()
ax.gridlines()

pcm = ax.pcolormesh(
    MLON, MLAT, delta_B_grid, transform=ccrs.PlateCarree(), cmap="RdBu_r"
)
cbar = plt.colorbar(pcm, ax=ax, orientation="vertical", label="扰动磁场 (nT)")

ax.plot(
    mlon_data, mlat_data, color="black", linewidth=0.5, transform=ccrs.PlateCarree()
)
ax.set_title("图2：扰动磁场的空间分布图（极区）")

plt.show()
