from pathlib import Path

import matplotlib.pyplot as plt

from utils import orbit
from core import zh1

from get_1d_file_names import get_1d_file_names

data_dir_path = Path(
    r"V:\aw\zh1\efd\ulf\2a\20210401_20210630"
)  # 数据大，所以不放进项目

day = "20210401"
file_names = get_1d_file_names(data_dir_path, condition=day)
file_paths = [data_dir_path / file_name for file_name in file_names]

lons_list = []
lats_list = []
for file_name,file_path in zip(file_names,file_paths):
    efd = zh1.EFD(file_path)
    df = efd.df1c
    lats = df["GEO_LAT"].values
    lons = df["GEO_LON"].values

    orbit_zh1 = orbit.OrbitZh1(file_name)
    indicator = orbit_zh1.indicator
    indices = orbit.get_nor_sou_split_indices_zh1(lats, indicator)

    nor_slice = slice(*indices[0])
    lons_list.append(lons[nor_slice])
    lats_list.append(lats[nor_slice])

# success call
orbit.orbits_hemisphere_projection(
    lons_list, lats_list, proj_method="NorthPolarStereo"
)
plt.show()