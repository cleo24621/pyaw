import os.path

import pandas as pd
from matplotlib import pyplot as plt

from pyaw.configs import ProjectConfigs
from pyaw.utils import orbit
from pyaw.core import dmsp

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
    lons_nor_list, lats_nor_list, lons_sou_list, lats_sou_list
)
plt.show()