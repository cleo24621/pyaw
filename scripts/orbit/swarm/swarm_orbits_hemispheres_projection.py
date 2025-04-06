import os.path

import pandas as pd
from matplotlib import pyplot as plt

from pyaw.configs import ProjectConfigs
from pyaw.utils import orbit

# 主程序部分
fns = [
    "only_gdcoors_SW_OPER_MAGA_LR_1B_12727_20160229T235551_20160301T012924.pkl",
    "only_gdcoors_SW_OPER_MAGA_LR_1B_12728_20160301T012924_20160301T030258.pkl",
    "only_gdcoors_SW_OPER_MAGA_LR_1B_12729_20160301T030258_20160301T043631.pkl",
    "only_gdcoors_SW_OPER_MAGA_LR_1B_12730_20160301T043631_20160301T061005.pkl",
    "only_gdcoors_SW_OPER_MAGA_LR_1B_12731_20160301T061005_20160301T074338.pkl",
    "only_gdcoors_SW_OPER_MAGA_LR_1B_12732_20160301T074338_20160301T091712.pkl",
    "only_gdcoors_SW_OPER_MAGA_LR_1B_12733_20160301T091712_20160301T105045.pkl",
    "only_gdcoors_SW_OPER_MAGA_LR_1B_12734_20160301T105045_20160301T122419.pkl",
    "only_gdcoors_SW_OPER_MAGA_LR_1B_12735_20160301T122419_20160301T135752.pkl",
    "only_gdcoors_SW_OPER_MAGA_LR_1B_12736_20160301T135752_20160301T153125.pkl",
    "only_gdcoors_SW_OPER_MAGA_LR_1B_12737_20160301T153125_20160301T170459.pkl",
    "only_gdcoors_SW_OPER_MAGA_LR_1B_12738_20160301T170459_20160301T183832.pkl",
    "only_gdcoors_SW_OPER_MAGA_LR_1B_12739_20160301T183832_20160301T201206.pkl",
    "only_gdcoors_SW_OPER_MAGA_LR_1B_12740_20160301T201206_20160301T214539.pkl",
    "only_gdcoors_SW_OPER_MAGA_LR_1B_12741_20160301T214539_20160301T231913.pkl",
    "only_gdcoors_SW_OPER_MAGA_LR_1B_12742_20160301T231913_20160302T005246.pkl",
]
data_dir_path = ProjectConfigs.data_dir_path
file_paths = [os.path.join(data_dir_path, i) for i in fns]
lons_nor_list = []
lats_nor_list = []
lons_sou_list = []
lats_sou_list = []
for file_path in file_paths:
    df = pd.read_pickle(str(file_path))
    lats = df["Latitude"].values
    lons = df["Longitude"].values
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


plt.suptitle("SwarmA Multi-Track Northern and Southern Hemisphere Projection Map from 20160229T235551 to 20160302T005246")

#%% save
save_dir = r"G:\note\毕业论文\images"
save = True
if save:
    output_filename_png = f"SwarmA_Multi_Track_Northern_and_Southern_Hemisphere_Projection_Map_from_20160229T235551_to_20160302T005246.png"
    output_path = os.path.join(save_dir, output_filename_png)
    print(f"Saving figure to {output_filename_png} (300 DPI)")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")


#%% show
show = True
if show:
    plt.show()
