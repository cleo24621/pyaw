import pandas as pd
from matplotlib import pyplot as plt

from pyaw import configs
from pyaw.projection import get_nor_sou_split_indices_swarm_dmsp as get_idx
from pyaw.projection import orbit_hemisphere_projection

satellite = "Swarm"
file_name = "../../../data/Swarm/aux_/aux_SW_OPER_MAGA_LR_1B_12728_20160301T012924_20160301T030258.pkl"  # modify: different file
file_path = configs.DATA_DIR / file_name
df = pd.read_pickle(file_path)
lats = df["Latitude"].values
lons = df["Longitude"].values
indices = get_idx(lats)
hemisphere_slice = slice(*indices[0])
proj = "NorthPolarStereo"
hemisphere_lats = lats[hemisphere_slice]
hemisphere_lons = lons[hemisphere_slice]
# success call
fig, ax = orbit_hemisphere_projection(
    hemisphere_lons, hemisphere_lats, satellite=satellite, proj_method=proj
)
plt.show()
