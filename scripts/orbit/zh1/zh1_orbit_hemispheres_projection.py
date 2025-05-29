"""
plot zh1 norther and southern hemisphere satellite orbit projection figure.
"""

import os

from matplotlib import pyplot as plt

from core import zh1
from src.pyaw import ProjectConfigs
from utils import orbit

# 主程序部分
satellite = "zh1"
orbit_number = "17538"
st = '20210401_003440'
et = '20210401_010914'
data_dir_path = ProjectConfigs.data_dir_path
file_name = "CSES_01_EFD_1_L2A_A1_175380_20210401_003440_20210401_010914_000.h5"  # modify: different file
file_path = os.path.join(data_dir_path, file_name)

efd = zh1.EFD(file_path)
df1c = efd.df1c
lats = df1c["GEO_LAT"].values
lons = df1c["GEO_LON"].values

orbit_zh1 = orbit.OrbitZh1(file_name)
indicator = orbit_zh1.indicator
indices = orbit.get_nor_sou_split_indices_zh1(lats, indicator)
northern_slice = slice(*indices[0])
southern_slice = slice(*indices[1])

lons_north = lons[northern_slice]
lats_north = lats[northern_slice]

lons_south = lons[southern_slice]
lats_south = lats[southern_slice]

# success call
orbit.orbit_hemispheres_projection(
    lons_north, lats_north, lons_south, lats_south, satellite
)
plt.show()