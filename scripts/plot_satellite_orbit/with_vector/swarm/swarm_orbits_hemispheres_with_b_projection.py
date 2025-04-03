"""
绘制swarm多轨扰动磁场、电场投影
"""

import os.path
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from pyaw.configs import ProjectConfigs
from pyaw.utils import orbit
from pyaw.utils.other import get_3arrs

orbit_num_list = list(range(12727, 12743))
file_names_mag = [
    "SW_OPER_MAGA_LR_1B_12727_20160229T235551_20160301T012924.pkl",
    "SW_OPER_MAGA_LR_1B_12728_20160301T012924_20160301T030258.pkl",
    "SW_OPER_MAGA_LR_1B_12729_20160301T030258_20160301T043631.pkl",
    "SW_OPER_MAGA_LR_1B_12730_20160301T043631_20160301T061005.pkl",
    "SW_OPER_MAGA_LR_1B_12731_20160301T061005_20160301T074338.pkl",
    "SW_OPER_MAGA_LR_1B_12732_20160301T074338_20160301T091712.pkl",
    "SW_OPER_MAGA_LR_1B_12733_20160301T091712_20160301T105045.pkl",
    "SW_OPER_MAGA_LR_1B_12734_20160301T105045_20160301T122419.pkl",
    "SW_OPER_MAGA_LR_1B_12735_20160301T122419_20160301T135752.pkl",
    "SW_OPER_MAGA_LR_1B_12736_20160301T135752_20160301T153125.pkl",
    "SW_OPER_MAGA_LR_1B_12737_20160301T153125_20160301T170459.pkl",
    "SW_OPER_MAGA_LR_1B_12738_20160301T170459_20160301T183832.pkl",
    "SW_OPER_MAGA_LR_1B_12739_20160301T183832_20160301T201206.pkl",
    "SW_OPER_MAGA_LR_1B_12740_20160301T201206_20160301T214539.pkl",
    "SW_OPER_MAGA_LR_1B_12741_20160301T214539_20160301T231913.pkl",
    "SW_OPER_MAGA_LR_1B_12742_20160301T231913_20160302T005246.pkl",
]

file_names_igrf = [
    "IGRF_SW_OPER_MAGA_LR_1B_12727_20160229T235551_20160301T012924.pkl",
    "IGRF_SW_OPER_MAGA_LR_1B_12728_20160301T012924_20160301T030258.pkl",
    "IGRF_SW_OPER_MAGA_LR_1B_12729_20160301T030258_20160301T043631.pkl",
    "IGRF_SW_OPER_MAGA_LR_1B_12730_20160301T043631_20160301T061005.pkl",
    "IGRF_SW_OPER_MAGA_LR_1B_12731_20160301T061005_20160301T074338.pkl",
    "IGRF_SW_OPER_MAGA_LR_1B_12732_20160301T074338_20160301T091712.pkl",
    "IGRF_SW_OPER_MAGA_LR_1B_12733_20160301T091712_20160301T105045.pkl",
    "IGRF_SW_OPER_MAGA_LR_1B_12734_20160301T105045_20160301T122419.pkl",
    "IGRF_SW_OPER_MAGA_LR_1B_12735_20160301T122419_20160301T135752.pkl",
    "IGRF_SW_OPER_MAGA_LR_1B_12736_20160301T135752_20160301T153125.pkl",
    "IGRF_SW_OPER_MAGA_LR_1B_12737_20160301T153125_20160301T170459.pkl",
    "IGRF_SW_OPER_MAGA_LR_1B_12738_20160301T170459_20160301T183832.pkl",
    "IGRF_SW_OPER_MAGA_LR_1B_12739_20160301T183832_20160301T201206.pkl",
    "IGRF_SW_OPER_MAGA_LR_1B_12740_20160301T201206_20160301T214539.pkl",
    "IGRF_SW_OPER_MAGA_LR_1B_12741_20160301T214539_20160301T231913.pkl",
    "IGRF_SW_OPER_MAGA_LR_1B_12742_20160301T231913_20160302T005246.pkl",
]

# for aux static info
file_names_aux = [
    "aux_SW_OPER_MAGA_LR_1B_12727_20160229T235551_20160301T012924.pkl",
    "aux_SW_OPER_MAGA_LR_1B_12728_20160301T012924_20160301T030258.pkl",
    "aux_SW_OPER_MAGA_LR_1B_12729_20160301T030258_20160301T043631.pkl",
    "aux_SW_OPER_MAGA_LR_1B_12730_20160301T043631_20160301T061005.pkl",
    "aux_SW_OPER_MAGA_LR_1B_12731_20160301T061005_20160301T074338.pkl",
    "aux_SW_OPER_MAGA_LR_1B_12732_20160301T074338_20160301T091712.pkl",
    "aux_SW_OPER_MAGA_LR_1B_12733_20160301T091712_20160301T105045.pkl",
    "aux_SW_OPER_MAGA_LR_1B_12734_20160301T105045_20160301T122419.pkl",
    "aux_SW_OPER_MAGA_LR_1B_12735_20160301T122419_20160301T135752.pkl",
    "aux_SW_OPER_MAGA_LR_1B_12736_20160301T135752_20160301T153125.pkl",
    "aux_SW_OPER_MAGA_LR_1B_12737_20160301T153125_20160301T170459.pkl",
    "aux_SW_OPER_MAGA_LR_1B_12738_20160301T170459_20160301T183832.pkl",
    "aux_SW_OPER_MAGA_LR_1B_12739_20160301T183832_20160301T201206.pkl",
    "aux_SW_OPER_MAGA_LR_1B_12740_20160301T201206_20160301T214539.pkl",
    "aux_SW_OPER_MAGA_LR_1B_12741_20160301T214539_20160301T231913.pkl",
    "aux_SW_OPER_MAGA_LR_1B_12742_20160301T231913_20160302T005246.pkl",
]

data_dir_path = ProjectConfigs.data_dir_path

file_paths_measurement = [
    os.path.join(data_dir_path, file_name_measurement)
    for file_name_measurement in file_names_mag
]
file_paths_igrf = [
    os.path.join(data_dir_path, file_name_igrf) for file_name_igrf in file_names_igrf
]
# for static info
file_paths_aux = [
    os.path.join(data_dir_path, file_name_aux) for file_name_aux in file_names_aux
]

lons_nor_list = []
lats_nor_list = []
delta_B_E_nor_list = []
delta_B_N_nor_list = []
lons_sou_list = []
lats_sou_list = []
delta_B_E_sou_list = []
delta_B_N_sou_list = []


# b_nor_mean_list = []
# b_sou_mean_list = []

# plot for paper
def plot_for_paper(datetimes, b1, b2, b3, orbit_number, if_save=False, save_dir=r"G:\note\毕业论文\images"):
    """

    Args:
        datetimes:
        b1:e
        b2: n
        b3: c
        orbit_number:
        if_save:
        save_dir:

    Returns:

    """
    dt = datetime.fromisoformat(str(datetimes[0]))
    st = dt.strftime("%Y%m%dT%H%M%S")
    dt = datetime.fromisoformat(str(datetimes[-1]))
    et = dt.strftime("%Y%m%dT%H%M%S")

    plt.style.use(
        "seaborn-v0_8-paper"
    )
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    # row1
    ax1.plot(datetimes, b1, label=r"$\Delta B_{East}$")
    ax1.grid(True,
            which='major',
            axis='both',
            color='lightgray',  # 使用浅灰色，不突兀
            linestyle='-',  # 实线通常最清晰
            linewidth=0.6,  # 较细的线条
            alpha=0.8  # 可以稍微透明一点
            )
    ax1.set_ylabel("Magnetic Flux Density (nT)")
    ax1.set_title("East component of Disturb Magnetic Field")
    ax1.legend(loc='best')

    # row2
    ax2.plot(datetimes, b2, label=r"$\Delta B_{North}$")
    ax2.grid(True,
            which='major',
            axis='both',
            color='lightgray',  # 使用浅灰色，不突兀
            linestyle='-',  # 实线通常最清晰
            linewidth=0.6,  # 较细的线条
            alpha=0.8  # 可以稍微透明一点
            )
    ax2.set_ylabel("Magnetic Flux Density (nT)")
    ax2.set_title("North component of Disturb Magnetic Field")
    ax2.legend(loc='best')

    # row3
    ax3.plot(datetimes, b3, label=r"$\Delta B_{Centric}$")
    ax3.grid(True,
            which='major',
            axis='both',
            color='lightgray',  # 使用浅灰色，不突兀
            linestyle='-',  # 实线通常最清晰
            linewidth=0.6,  # 较细的线条
            alpha=0.8  # 可以稍微透明一点
            )
    ax3.set_xlabel("Time UTC")
    ax3.set_ylabel("Magnetic Flux Density (nT)")
    ax3.set_title("Centric Downward component of Disturb Magnetic Field")
    ax3.legend(loc='best')

    plt.suptitle(
        f"Three components of Disturb Magnetic Field in NEC Frame\nOrbit Number is {orbit_number}, From {st} to {et}")

    if if_save:
        output_filename_png = f"Three_components_of_Magnetic_Field_in_NEC_Frame_Orbit_Number_{orbit_number}_{st}_{et}.png"
        output_path = os.path.join(save_dir, output_filename_png)
        print(f"Saving figure to {output_filename_png} (300 DPI)")
        fig.savefig(output_path, dpi=300, bbox_inches="tight")


for orbit_num, file_path_measurement, file_path_igrf, file_path_aux in zip(
        orbit_num_list, file_paths_measurement, file_paths_igrf, file_paths_aux
):
    df_measurement = pd.read_pickle(str(file_path_measurement))
    df_igrf = pd.read_pickle(str(file_path_igrf))
    lats = df_measurement["Latitude"].values
    lons = df_measurement["Longitude"].values

    B_N, B_E, B_C = get_3arrs(df_measurement["B_NEC"].values)
    B_N_IGRF, B_E_IGRF, B_C_IGRF = get_3arrs(df_igrf["B_NEC_IGRF"].values)
    delta_B_E = B_E - B_E_IGRF
    delta_B_N = B_N - B_N_IGRF
    delta_B_C = B_C - B_C_IGRF

    print(f"plot {orbit_num} 3 components of Disturb Magnetic Field")
    plot_for_paper(datetimes=df_measurement.index.values, b1=delta_B_E, b2=delta_B_N, b3=delta_B_C,
                   orbit_number=orbit_num, if_save=True)
    print("end plot")

    # plot for paper

    indices = orbit.get_nor_sou_split_indices_swarm_dmsp(lats)
    nor_slice = slice(*indices[0])
    sou_slice = slice(*indices[1])

    lons_nor_list.append(lons[nor_slice])
    lats_nor_list.append(lats[nor_slice])
    delta_B_E_nor_list.append(delta_B_E[nor_slice])
    delta_B_N_nor_list.append(delta_B_N[nor_slice])

    lons_sou_list.append(lons[sou_slice])
    lats_sou_list.append(lats[sou_slice])
    delta_B_E_sou_list.append(delta_B_E[sou_slice])
    delta_B_N_sou_list.append(delta_B_N[sou_slice])

    # print some static info
    b_nor = np.sqrt(delta_B_E[nor_slice] ** 2 + delta_B_N[nor_slice] ** 2)
    b_sou = np.sqrt(delta_B_E[sou_slice] ** 2 + delta_B_N[sou_slice] ** 2)
    # b_nor_mean_list.append(np.mean(b_nor))
    # b_sou_mean_list.append(np.mean(b_sou))

    nor_max_idx = np.argmax(b_nor)
    sou_max_idx = np.argmax(b_sou)

    df_aux = pd.read_pickle(file_path_aux)
    mlats = df_aux['QDLat'].values
    mlts = df_aux['MLT'].values

    mlats_nor = mlats[nor_slice]
    mlats_sou = mlats[sou_slice]

    mlts_nor = mlts[nor_slice]
    mlts_sou = mlts[sou_slice]

    print(f"orbit_num: {orbit_num}")
    print("north---\n")
    print(f"b_nor_max: {b_nor[nor_max_idx]}",
          f"corresponding lat: {lats[nor_slice][nor_max_idx]}",
          f"corresponding mlat: {mlats_nor[nor_max_idx]}",
          f"corresponding mlt: {mlts_nor[nor_max_idx]}",
          f"b_nor_mean: {np.mean(b_nor)}")
    print("south---\n")
    print(f"b_sou_max: {b_sou[sou_max_idx]}",
          f"corresponding lat: {lats[sou_slice][sou_max_idx]}",
          f"corresponding mlat: {mlats_sou[sou_max_idx]}",
          f"corresponding mlt: {mlts_sou[sou_max_idx]}",
          f"b_sou_mean: {np.mean(b_sou)}"
          )
    print("\n")
    print("\n")
    print("---")

# success call
fig, axes = orbit.plot_dual_hemisphere_orbits(
    lons_nor_list,
    lats_nor_list,
    delta_B_E_nor_list,
    delta_B_N_nor_list,
    lons_sou_list,
    lats_sou_list,
    delta_B_E_sou_list,
    delta_B_N_sou_list,
    quiver_key_magnitude=100,
    if_cfeature=True,
    main_title="The Hemisphere Projection of SwarmA Disturb Magnetic Field from 20160229T235551 to 20160302T005246",
    vector_color="crimson",
    quiver_scale=10000.0,
    quiver_key_X=0.95,
    quiver_key_Y=0.05,
    vector_units_label="nT"
)

# %% save
save_dir = r"G:\note\毕业论文\images"
save = False
if save:
    output_filename_png = f"The_Hemisphere_Projection_of_SwarmA_Disturb_Magnetic_Field_from_20160229T235551_to_20160302T005246.png"
    output_path = os.path.join(save_dir, output_filename_png)
    print(f"Saving figure to {output_filename_png} (300 DPI)")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")