import os

from matplotlib import pyplot as plt

from src.pyaw import ProjectConfigs
from utils import orbit
from core import zh1

satellite = "zh1"
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
hemisphere_slice = slice(*indices[0])
proj = "NorthPolarStereo"
hemisphere_lats = lats[hemisphere_slice]
hemisphere_lons = lons[hemisphere_slice]
# success call
fig, ax = orbit.orbit_hemisphere_projection(
    hemisphere_lons, hemisphere_lats, satellite=satellite, proj_method=proj
)
plt.show()