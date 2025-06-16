import os

from matplotlib import pyplot as plt
from utils import orbit

from core import dmsp
from src.pyaw import ProjectConfigs

satellite = "dmsp"
data_dir_path = ProjectConfigs.data_dir_path
file_name = "../../../data/DMSP/dmsp-f16_ssies-3_thermal-plasma_201401010137_v01.cdf"  # modify: different file
file_path = os.path.join(data_dir_path, file_name)

ssies3 = dmsp.SPDF.SSIES3(file_path)
df = ssies3.original_df
lats = df["glat"].values
lons = df["glon"].values
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
