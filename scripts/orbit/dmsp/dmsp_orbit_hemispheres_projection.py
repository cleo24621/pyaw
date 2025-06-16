import os

from matplotlib import pyplot as plt
from utils import orbit

from core import dmsp
from src.pyaw import ProjectConfigs

# 主程序部分
satellite = "dmsp"
data_dir_path = ProjectConfigs.data_dir_path
file_name = "../../../data/DMSP/dmsp-f16_ssies-3_thermal-plasma_201401010137_v01.cdf"  # modify: different file
file_path = os.path.join(data_dir_path, file_name)

ssies3 = dmsp.SPDF.SSIES3(file_path)
df = ssies3.original_df
lats = df["glat"].values
lons = df["glon"].values
indices = orbit.get_nor_sou_split_indices_swarm_dmsp(lats)
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
