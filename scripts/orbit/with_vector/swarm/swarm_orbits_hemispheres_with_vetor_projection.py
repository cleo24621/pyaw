import os.path

import pandas as pd
from matplotlib import pyplot as plt

from src.pyaw import ProjectConfigs
from utils import orbit
from src.pyaw import get_3arrs, normalize_array

# 主程序部分
# fns = [
#     "only_gdcoors_SW_OPER_MAGA_LR_1B_12727_20160229T235551_20160301T012924.pkl",
#     "only_gdcoors_SW_OPER_MAGA_LR_1B_12728_20160301T012924_20160301T030258.pkl",
#     "only_gdcoors_SW_OPER_MAGA_LR_1B_12729_20160301T030258_20160301T043631.pkl",
#     # "only_gdcoors_SW_OPER_MAGA_LR_1B_12730_20160301T043631_20160301T061005.pkl",
#     # "only_gdcoors_SW_OPER_MAGA_LR_1B_12731_20160301T061005_20160301T074338.pkl",
#     # "only_gdcoors_SW_OPER_MAGA_LR_1B_12732_20160301T074338_20160301T091712.pkl",
#     # "only_gdcoors_SW_OPER_MAGA_LR_1B_12733_20160301T091712_20160301T105045.pkl",
#     # "only_gdcoors_SW_OPER_MAGA_LR_1B_12734_20160301T105045_20160301T122419.pkl",
#     # "only_gdcoors_SW_OPER_MAGA_LR_1B_12735_20160301T122419_20160301T135752.pkl",
#     # "only_gdcoors_SW_OPER_MAGA_LR_1B_12736_20160301T135752_20160301T153125.pkl",
#     # "only_gdcoors_SW_OPER_MAGA_LR_1B_12737_20160301T153125_20160301T170459.pkl",
#     # "only_gdcoors_SW_OPER_MAGA_LR_1B_12738_20160301T170459_20160301T183832.pkl",
#     # "only_gdcoors_SW_OPER_MAGA_LR_1B_12739_20160301T183832_20160301T201206.pkl",
#     # "only_gdcoors_SW_OPER_MAGA_LR_1B_12740_20160301T201206_20160301T214539.pkl",
#     # "only_gdcoors_SW_OPER_MAGA_LR_1B_12741_20160301T214539_20160301T231913.pkl",
#     # "only_gdcoors_SW_OPER_MAGA_LR_1B_12742_20160301T231913_20160302T005246.pkl",
# ]

file_names_measurement = ["SW_OPER_MAGA_LR_1B_12728_20160301T012924_20160301T030258.pkl",
                          "SW_OPER_MAGA_LR_1B_12729_20160301T030258_20160301T043631.pkl",
                          "SW_OPER_MAGA_LR_1B_12730_20160301T043631_20160301T061005.pkl",]
file_names_igrf = ['IGRF_SW_OPER_MAGA_LR_1B_12728_20160301T012924_20160301T030258.pkl',
                   'IGRF_SW_OPER_MAGA_LR_1B_12729_20160301T030258_20160301T043631.pkl',
                   'IGRF_SW_OPER_MAGA_LR_1B_12730_20160301T043631_20160301T061005.pkl']


data_dir_path = ProjectConfigs.data_dir_path
# file_paths = [os.path.join(data_dir_path, i) for i in fns]
file_paths_measurement = [os.path.join(data_dir_path,file_name_measurement) for file_name_measurement in file_names_measurement]
file_paths_igrf = [os.path.join(data_dir_path,file_name_igrf) for file_name_igrf in file_names_igrf]

lons_nor_list = []
lats_nor_list = []
delta_B_E_nor_list = []
delta_B_N_nor_list = []
lons_sou_list = []
lats_sou_list = []
delta_B_E_sou_list = []
delta_B_N_sou_list = []
for file_path_measurement,file_path_igrf in zip(file_paths_measurement,file_paths_igrf):
    df_measurement = pd.read_pickle(str(file_path_measurement))
    df_igrf = pd.read_pickle(str(file_path_igrf))
    lats = df_measurement["Latitude"].values
    lons = df_measurement["Longitude"].values

    B_N,B_E,_ = get_3arrs(df_measurement['B_NEC'].values)
    B_N_IGRF,B_E_IGRF,_ = get_3arrs(df_igrf['B_NEC_IGRF'].values)
    delta_B_E = B_E - B_E_IGRF
    delta_B_N = B_N - B_N_IGRF
    # normalize delta_B for plot
    delta_B_E_normalized = normalize_array(delta_B_E,target_min=-1)
    delta_B_N_normalized = normalize_array(delta_B_N,target_min=-1)

    indices = orbit.get_nor_sou_split_indices_swarm_dmsp(lats)
    nor_slice = slice(*indices[0])
    sou_slice = slice(*indices[1])

    lons_nor_list.append(lons[nor_slice])
    lats_nor_list.append(lats[nor_slice])
    delta_B_E_nor_list.append(delta_B_E_normalized[nor_slice])
    delta_B_N_nor_list.append(delta_B_N_normalized[nor_slice])

    lons_sou_list.append(lons[sou_slice])
    lats_sou_list.append(lats[sou_slice])
    delta_B_E_sou_list.append(delta_B_E_normalized[sou_slice])
    delta_B_N_sou_list.append(delta_B_N_normalized[sou_slice])


# success call
fig,ax = orbit.orbits_hemisphere_with_vector_projection(
    lons_nor_list, lats_nor_list,delta_B_E_nor_list,delta_B_N_nor_list,proj_method='NorthPolarStereo',step=10
)
plt.suptitle("Vector SwarmA Multi-Track Northern and Southern Hemisphere Projection Map from 20160229T235551 to 20160302T005246")

#%% save
save_dir = r"G:\note\毕业论文\images"
save = True
if save:
    output_filename_png = f"SwarmA_Multi_Track_Northern_and_Southern_Hemisphere_Projection_Map_from_20160229T235551_to_20160302T005246.png"
    output_path = os.path.join(save_dir, output_filename_png)
    print(f"Saving figure to {output_filename_png} (300 DPI)")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")


#%% show
show = True
if show:
    plt.show()