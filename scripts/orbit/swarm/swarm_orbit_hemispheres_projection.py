import os

import pandas as pd
from matplotlib import pyplot as plt

from pyaw.configs import ProjectConfigs
from pyaw.utils import orbit

# 主程序部分
satellite = "Swarm"
spacecraft = "A"
orbit_number = 12727
st = "20160229T235551"
et = "20160301T012924"
data_dir_path = ProjectConfigs.data_dir_path
file_name = "aux_SW_OPER_MAGA_LR_1B_12728_20160301T012924_20160301T030258.pkl"  # modify: different file
file_path = os.path.join(data_dir_path, file_name)
df = pd.read_pickle(file_path)
lats = df["Latitude"].values
lons = df["Longitude"].values
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