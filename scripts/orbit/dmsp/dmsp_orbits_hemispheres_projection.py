import os.path

import pandas as pd
from matplotlib import pyplot as plt

from pyaw.configs import ProjectConfigs
from pyaw.utils import orbit
from pyaw.core import dmsp

# 主程序部分
data_dir_path = ProjectConfigs.data_dir_path
day = '20140101'
serial_number = '16'
file_names = [f for f in os.listdir(data_dir_path) if "ssies-3" in f and day in f and f.endswith('.cdf')]
file_paths = [os.path.join(data_dir_path,file_name) for file_name in file_names]
lons_nor_list = []
lats_nor_list = []
lons_sou_list = []
lats_sou_list = []
for file_path in file_paths:
    ssies3 = dmsp.SPDF.SSIES3(file_path)
    df = ssies3.original_df
    lats = df["glat"].values
    lons = df["glon"].values
    indices = orbit.get_nor_sou_split_indices_swarm_dmsp(lats)
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
plt.suptitle("DMSP F16 Multi-Track Northern and Southern Hemisphere Projection Map on 20140101")

#%% save
save_dir = r"G:\note\毕业论文\images"
save = True
if save:
    output_filename_png = f"DMSP_F16_Multi-Track_Northern_and_Southern_Hemisphere_Projection_Map_on_20140101.png"
    output_path = os.path.join(save_dir, output_filename_png)
    print(f"Saving figure to {output_filename_png} (300 DPI)")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")


#%% show
show = True
if show:
    plt.show()