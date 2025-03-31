import os.path

import pandas as pd
from matplotlib import pyplot as plt

from pyaw.configs import ProjectConfigs
from pyaw.utils import orbit
from utils import coordinate
from utils.other import OutlierData

file_names_tct16 = [
    "SW_EXPT_EFIA_TCT16_12727_20160229T235551_20160301T012924.pkl",
    "SW_EXPT_EFIA_TCT16_12728_20160301T012924_20160301T030258.pkl",
    "SW_EXPT_EFIA_TCT16_12729_20160301T030258_20160301T043631.pkl",
    "SW_EXPT_EFIA_TCT16_12730_20160301T043631_20160301T061005.pkl",
    "SW_EXPT_EFIA_TCT16_12731_20160301T061005_20160301T074338.pkl",
    "SW_EXPT_EFIA_TCT16_12732_20160301T074338_20160301T091712.pkl",
    "SW_EXPT_EFIA_TCT16_12733_20160301T091712_20160301T105045.pkl",
    "SW_EXPT_EFIA_TCT16_12734_20160301T105045_20160301T122419.pkl",
    "SW_EXPT_EFIA_TCT16_12735_20160301T122419_20160301T135752.pkl",
    "SW_EXPT_EFIA_TCT16_12736_20160301T135752_20160301T153125.pkl",
    "SW_EXPT_EFIA_TCT16_12737_20160301T153125_20160301T170459.pkl",
    "SW_EXPT_EFIA_TCT16_12738_20160301T170459_20160301T183832.pkl",
    "SW_EXPT_EFIA_TCT16_12739_20160301T183832_20160301T201206.pkl",
    "SW_EXPT_EFIA_TCT16_12740_20160301T201206_20160301T214539.pkl",
    "SW_EXPT_EFIA_TCT16_12741_20160301T214539_20160301T231913.pkl",
    "SW_EXPT_EFIA_TCT16_12742_20160301T231913_20160302T005246.pkl",
]
data_dir_path = ProjectConfigs.data_dir_path
file_paths = [os.path.join(data_dir_path, file_name) for file_name in file_names_tct16]

lons_nor_list = []
lats_nor_list = []
E_E_nor_list = []
E_N_nor_list = []
lons_sou_list = []
lats_sou_list = []
E_E_sou_list = []
E_N_sou_list = []
for file_path in file_paths:
    df = pd.read_pickle(str(file_path))
    if df.empty:
        continue
    Ehx = df["Ehx"].values
    Ehy = df["Ehy"].values
    # outlier
    Ehx = OutlierData.set_outliers_nan_std(Ehx, 1, print_=True)
    Ehy = OutlierData.set_outliers_nan_std(Ehy, 1, print_=True)

    VsatN = df["VsatN"].values
    VsatE = df["VsatE"].values

    # 计算旋转矩阵
    rotmat_nec2sc, rotmat_sc2nec = coordinate.NEC2SCandSC2NEC.get_rotmat_nec2sc_sc2nec(
        VsatN, VsatE
    )
    E_north, E_east = coordinate.NEC2SCandSC2NEC.do_rotation(
        -Ehx, -Ehy, rotmat_sc2nec
    )  # todo: use '-'

    lats = df["Latitude"].values
    lons = df["Longitude"].values

    indices = orbit.get_nor_sou_split_indices_swarm_dmsp(lats)
    nor_slice = slice(*indices[0])
    sou_slice = slice(*indices[1])

    lons_nor_list.append(lons[nor_slice])
    lats_nor_list.append(lats[nor_slice])
    E_E_nor_list.append(E_east[nor_slice])
    E_N_nor_list.append(E_north[nor_slice])

    lons_sou_list.append(lons[sou_slice])
    lats_sou_list.append(lats[sou_slice])
    E_E_sou_list.append(E_east[sou_slice])
    E_N_sou_list.append(E_north[sou_slice])


# success call
orbit.plot_dual_hemisphere_orbits(
    lons_nor_list,
    lats_nor_list,
    E_E_nor_list,
    E_N_nor_list,
    lons_sou_list,
    lats_sou_list,
    E_E_sou_list,
    E_N_sou_list,
    quiver_key_magnitude=100,
    if_cfeature=False,
    main_title="The Hemisphere Projection of SwarmA Electric Field from 20160229T235551 to 20160302T005246",
    vector_color="crimson",
    quiver_scale=10000.0,
    quiver_key_X=0.95,
    quiver_key_Y=0.05,
)
plt.show()

# conclusion
# 效果不好