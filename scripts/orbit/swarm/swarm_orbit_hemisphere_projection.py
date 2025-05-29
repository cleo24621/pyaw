import os

import pandas as pd
from matplotlib import pyplot as plt

from src.pyaw import ProjectConfigs
from utils import orbit

satellite = "Swarm"
data_dir_path = ProjectConfigs.data_dir_path
file_name = "aux_SW_OPER_MAGA_LR_1B_12728_20160301T012924_20160301T030258.pkl"  # modify: different file
file_path = os.path.join(data_dir_path, file_name)
df = pd.read_pickle(file_path)
lats = df["Latitude"].values
lons = df["Longitude"].values
indices = orbit.get_nor_sou_split_indices_swarm_dmsp(lats)
hemisphere_slice = slice(*indices[0])
proj = "NorthPolarStereo"
hemisphere_lats = lats[hemisphere_slice]
hemisphere_lons = lons[hemisphere_slice]
# success call
fig, ax = orbit.orbit_hemisphere_projection(
    hemisphere_lons, hemisphere_lats, satellite=satellite, proj_method=proj
)
plt.show()