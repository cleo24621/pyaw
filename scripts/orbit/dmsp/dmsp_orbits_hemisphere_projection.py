import os.path

from matplotlib import pyplot as plt

from src.pyaw import ProjectConfigs
from utils import orbit
from core import dmsp

# 主程序部分
data_dir_path = ProjectConfigs.data_dir_path
day = '20140101'
serial_number = '16'
file_names = [f for f in os.listdir(data_dir_path) if "ssies-3" in f and day in f and f.endswith('.cdf')]
file_paths = [os.path.join(data_dir_path,file_name) for file_name in file_names]
lons_list = []
lats_list = []
for file_path in file_paths:
    ssies3 = dmsp.SPDF.SSIES3(file_path)
    df = ssies3.original_df
    lats = df["glat"].values
    lons = df["glon"].values
    indices = orbit.get_nor_sou_split_indices_swarm_dmsp(lats)
    nor_slice = slice(*indices[0])
    lons_list.append(lons[nor_slice])
    lats_list.append(lats[nor_slice])
# success call
orbit.orbits_hemisphere_projection(
    lons_list, lats_list, proj_method="NorthPolarStereo"
)
plt.show()