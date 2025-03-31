from pathlib import Path

import matplotlib.pyplot as plt

from pyaw.utils import orbit
from pyaw.core import zh1

from get_1d_file_names import get_1d_file_names

data_dir_path = Path(
    r"V:\aw\zh1\efd\ulf\2a\20210401_20210630"
)  # 数据大，所以不放进项目

day = "20210401"
file_names = get_1d_file_names(data_dir_path, condition=day)
file_paths = [data_dir_path / file_name for file_name in file_names]

lons_nor_list = []
lats_nor_list = []
lons_sou_list = []
lats_sou_list = []
for file_name,file_path in zip(file_names,file_paths):
    efd = zh1.EFD(file_path)
    df = efd.df1c
    lats = df["GEO_LAT"].values
    lons = df["GEO_LON"].values

    orbit_zh1 = orbit.OrbitZh1(file_name)
    indicator = orbit_zh1.indicator
    indices = orbit_zh1.get_nor_sou_split_indices(lats)

    nor_slice = slice(*indices[0])
    sou_slice = slice(*indices[1])
    lons_nor_list.append(lons[nor_slice])
    lats_nor_list.append(lats[nor_slice])
    lons_sou_list.append(lons[sou_slice])
    lats_sou_list.append(lats[sou_slice])

# success call
orbit.orbits_hemispheres_projection(
    lons_nor_list, lats_nor_list, lons_sou_list, lats_sou_list,if_cfeature=True
)

plt.suptitle("ZH-1 Multi-Track Northern and Southern Hemisphere Projection Map on 2021-04-01")
plt.show()