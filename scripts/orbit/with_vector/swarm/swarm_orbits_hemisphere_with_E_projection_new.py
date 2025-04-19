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
from pyaw.utils.other import get_3arrs, OutlierData, interpolate_missing
from utils import coordinate

orbit_num_list = [12728,12729,12738,12739,12740,12741,12742,12743,12744,12753,12753,12754,12755,12756,12757,12758,12759]
file_names_tct6 = [
    "SW_EXPT_EFIA_TCT16_12728_20160301T012924_20160301T030258.pkl",
    "SW_EXPT_EFIA_TCT16_12729_20160301T030258_20160301T043631.pkl",
    "SW_EXPT_EFIA_TCT16_12738_20160301T170459_20160301T183832.pkl",
    "SW_EXPT_EFIA_TCT16_12739_20160301T183832_20160301T201206.pkl",
    "SW_EXPT_EFIA_TCT16_12740_20160301T201206_20160301T214539.pkl",
    "SW_EXPT_EFIA_TCT16_12741_20160301T214539_20160301T231913.pkl",
    "SW_EXPT_EFIA_TCT16_12742_20160301T231913_20160302T005246.pkl",
    "SW_EXPT_EFIA_TCT16_12743_20160302T005246_20160302T022620.pkl",
    "SW_EXPT_EFIA_TCT16_12744_20160302T022620_20160302T035953.pkl",
    "SW_EXPT_EFIA_TCT16_12753_20160302T162820_20160302T180154.pkl",
    "SW_EXPT_EFIA_TCT16_12753_20160302T162820_20160302T180154.pkl",
    "SW_EXPT_EFIA_TCT16_12754_20160302T180154_20160302T193527.pkl",
    "SW_EXPT_EFIA_TCT16_12755_20160302T193527_20160302T210901.pkl",
    "SW_EXPT_EFIA_TCT16_12756_20160302T210901_20160302T224234.pkl",
    "SW_EXPT_EFIA_TCT16_12757_20160302T224234_20160303T001608.pkl",
    "SW_EXPT_EFIA_TCT16_12758_20160303T001608_20160303T014941.pkl",
    "SW_EXPT_EFIA_TCT16_12759_20160303T014941_20160303T032314.pkl",


]

# for aux static info
file_names_aux = [
    "aux_SW_EXPT_EFIA_TCT16_12728_20160301T012924_20160301T030258.pkl",
    "aux_SW_EXPT_EFIA_TCT16_12729_20160301T030258_20160301T043631.pkl",
    "aux_SW_EXPT_EFIA_TCT16_12738_20160301T170459_20160301T183832.pkl",
    "aux_SW_EXPT_EFIA_TCT16_12739_20160301T183832_20160301T201206.pkl",
    "aux_SW_EXPT_EFIA_TCT16_12740_20160301T201206_20160301T214539.pkl",
    "aux_SW_EXPT_EFIA_TCT16_12741_20160301T214539_20160301T231913.pkl",
    "aux_SW_EXPT_EFIA_TCT16_12742_20160301T231913_20160302T005246.pkl",
    "aux_SW_EXPT_EFIA_TCT16_12743_20160302T005246_20160302T022620.pkl",
    "aux_SW_EXPT_EFIA_TCT16_12744_20160302T022620_20160302T035953.pkl",
    "aux_SW_EXPT_EFIA_TCT16_12753_20160302T162820_20160302T180154.pkl",
    "aux_SW_EXPT_EFIA_TCT16_12754_20160302T180154_20160302T193527.pkl",
    "aux_SW_EXPT_EFIA_TCT16_12755_20160302T193527_20160302T210901.pkl",
    "aux_SW_EXPT_EFIA_TCT16_12756_20160302T210901_20160302T224234.pkl",
    "aux_SW_EXPT_EFIA_TCT16_12757_20160302T224234_20160303T001608.pkl",
    "aux_SW_EXPT_EFIA_TCT16_12758_20160303T001608_20160303T014941.pkl",
    "aux_SW_EXPT_EFIA_TCT16_12759_20160303T014941_20160303T032314.pkl",
]

data_dir_path = ProjectConfigs.data_dir_path

file_paths_measurement = [
    os.path.join(data_dir_path, file_name_measurement)
    for file_name_measurement in file_names_tct6
]

# for static info
file_paths_aux = [
    os.path.join(data_dir_path, file_name_aux) for file_name_aux in file_names_aux
]

lons_nor_list = []
lats_nor_list = []
E_east_nor_list = []
E_north_nor_list = []
lons_sou_list = []
lats_sou_list = []
E_east_sou_list = []
E_north_sou_list = []


# b_nor_mean_list = []
# b_sou_mean_list = []

# plot for paper
def plot_for_paper(datetimes, E1, E2, E3, orbit_number, if_save=False, save_dir=r"G:\note\毕业论文\images"):
    """

    Args:
        datetimes:
        E1:e
        E2: n
        E3: c
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
    ax1.plot(datetimes, E1, label=r"$E_{East}$")
    ax1.grid(True,
            which='major',
            axis='both',
            color='lightgray',  # 使用浅灰色，不突兀
            linestyle='-',  # 实线通常最清晰
            linewidth=0.6,  # 较细的线条
            alpha=0.8  # 可以稍微透明一点
            )
    ax1.set_ylabel("Electric Field Strength (mV/m)")
    ax1.set_title("East component of Electric Field")
    ax1.legend(loc='best')

    # row2
    ax2.plot(datetimes, E2, label=r"$E_{North}$")
    ax2.grid(True,
            which='major',
            axis='both',
            color='lightgray',  # 使用浅灰色，不突兀
            linestyle='-',  # 实线通常最清晰
            linewidth=0.6,  # 较细的线条
            alpha=0.8  # 可以稍微透明一点
            )
    ax2.set_ylabel("Electric Field Strength (mV/m)")
    ax2.set_title("North component of Electric Field")
    ax2.legend(loc='best')

    # row3
    ax3.plot(datetimes, E3, label=r"$E_{Centric}$")
    ax3.grid(True,
            which='major',
            axis='both',
            color='lightgray',  # 使用浅灰色，不突兀
            linestyle='-',  # 实线通常最清晰
            linewidth=0.6,  # 较细的线条
            alpha=0.8  # 可以稍微透明一点
            )
    ax3.set_xlabel("Time UTC")
    ax3.set_ylabel("Electric Field Strength (mV/m)")
    ax3.set_title("Centric Downward component of Electric Field")
    ax3.legend(loc='best')

    plt.suptitle(
        f"Three components of Electric Field in NEC Frame\nOrbit Number is {orbit_number}, From {st} to {et}")

    if if_save:
        output_filename_png = f"Three_components_of_Electric_Field_in_NEC_Frame_Orbit_Number_{orbit_number}_{st}_{et}.png"
        output_path = os.path.join(save_dir, output_filename_png)
        print(f"Saving figure to {output_filename_png} (300 DPI)")
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

#todo: may need modify the data process for better figures
output_filename = "orbit_log.txt"
with open(output_filename, 'w') as outfile:
    for orbit_num, file_path_measurement, file_path_aux in zip(
            orbit_num_list, file_paths_measurement, file_paths_aux
    ):
        df_measurement = pd.read_pickle(str(file_path_measurement))
        lats = df_measurement["Latitude"].values
        lons = df_measurement["Longitude"].values

        Ehx = df_measurement["Ehx"].values
        Ehx_outlier = OutlierData.set_outliers_nan_std(Ehx, 3, print_=True)

        Ehy = df_measurement["Ehy"].values
        Ehy_outlier = OutlierData.set_outliers_nan_std(Ehy, 3, print_=True)

        Ehz = df_measurement["Ehz"].values
        Ehz_outlier = OutlierData.set_outliers_nan_std(Ehz, 3, print_=True)

        VsatN = df_measurement["VsatN"].values
        VsatE = df_measurement["VsatE"].values

        rotmat_nec2sc, rotmat_sc2nec = coordinate.NEC2SCandSC2NEC.get_rotmat_nec2sc_sc2nec(
            VsatN, VsatE
        )
        E_north, E_east = coordinate.NEC2SCandSC2NEC.do_rotation(
            -Ehx_outlier, -Ehy_outlier, rotmat_sc2nec
        )


        print(f"plot {orbit_num} 3 components of Electric Field")
        plot_for_paper(datetimes=df_measurement.index.values, E1=E_north, E2=E_east, E3=Ehz_outlier,
                       orbit_number=orbit_num, if_save=True)
        print("end plot")

        # plot for paper

        indices = orbit.get_nor_sou_split_indices_swarm_dmsp(lats)
        nor_slice = slice(*indices[0])
        sou_slice = slice(*indices[1])

        lons_nor_list.append(lons[nor_slice])
        lats_nor_list.append(lats[nor_slice])
        E_east_nor_list.append(E_east[nor_slice])
        E_north_nor_list.append(E_north[nor_slice])

        lons_sou_list.append(lons[sou_slice])
        lats_sou_list.append(lats[sou_slice])
        E_east_sou_list.append(E_east[sou_slice])
        E_north_sou_list.append(E_north[sou_slice])

        # print some static info
        E_nor = np.sqrt(E_east[nor_slice] ** 2 + E_north[nor_slice] ** 2)
        E_sou = np.sqrt(E_east[sou_slice] ** 2 + E_north[sou_slice] ** 2)
        # b_nor_mean_list.append(np.mean(b_nor))
        # b_sou_mean_list.append(np.mean(b_sou))

        nor_max_idx = np.argmax(E_nor)
        sou_max_idx = np.argmax(E_sou)

        df_aux = pd.read_pickle(file_path_aux)
        mlats = df_aux['QDLat'].values
        mlts = df_aux['MLT'].values

        mlats_nor = mlats[nor_slice]
        mlats_sou = mlats[sou_slice]

        mlts_nor = mlts[nor_slice]
        mlts_sou = mlts[sou_slice]
        try:
            outfile.write(f"orbit_num: {orbit_num}\n")
            outfile.write("north---\n")
            outfile.write(f"E_nor_max: {E_nor[nor_max_idx]}\n")
            outfile.write(f"corresponding lat: {lats[nor_slice][nor_max_idx]}\n")
            outfile.write(f"corresponding mlat: {mlats_nor[nor_max_idx]}\n")
            outfile.write(f"corresponding mlt: {mlts_nor[nor_max_idx]}\n")
            outfile.write(f"E_nor_mean: {np.mean(E_nor)}\n")

            outfile.write("south---\n")
            outfile.write(f"E_sou_max: {E_sou[sou_max_idx]}\n")
            outfile.write(f"corresponding lat: {lats[sou_slice][sou_max_idx]}\n")
            outfile.write(f"corresponding mlat: {mlats_sou[sou_max_idx]}\n")
            outfile.write(f"corresponding mlt: {mlts_sou[sou_max_idx]}\n")
            outfile.write(f"E_sou_mean: {np.mean(E_sou)}\n")
        except:
            continue


        print(f"orbit_num: {orbit_num}")
        print("north---\n")
        print(f"E_nor_max: {E_nor[nor_max_idx]}",
              f"corresponding lat: {lats[nor_slice][nor_max_idx]}",
              f"corresponding mlat: {mlats_nor[nor_max_idx]}",
              f"corresponding mlt: {mlts_nor[nor_max_idx]}",
              f"E_nor_mean: {np.mean(E_nor)}")
        print("south---\n")
        print(f"E_sou_max: {E_sou[sou_max_idx]}",
              f"corresponding lat: {lats[sou_slice][sou_max_idx]}",
              f"corresponding mlat: {mlats_sou[sou_max_idx]}",
              f"corresponding mlt: {mlts_sou[sou_max_idx]}",
              f"E_sou_mean: {np.mean(E_sou)}"
              )
        print("\n")
        print("\n")
        print("---")

# success call
fig, axes = orbit.plot_dual_hemisphere_orbits(
    lons_nor_list,
    lats_nor_list,
    E_east_nor_list,
    E_north_nor_list,
    lons_sou_list,
    lats_sou_list,
    E_east_sou_list,
    E_north_sou_list,
    quiver_key_magnitude=100,
    if_cfeature=True,
    main_title="The Hemisphere Projection of SwarmA Electric Field",
    vector_color="crimson",
    quiver_scale=10000.0,
    quiver_key_X=0.95,
    quiver_key_Y=0.05,
    vector_units_label="mV/m"
)

# %% save
save_dir = r"G:\note\毕业论文\images"
save = False
if save:
    output_filename_png = f"The_Hemisphere_Projection_of_SwarmA_Electric_Field.png"
    output_path = os.path.join(save_dir, output_filename_png)
    print(f"Saving figure to {output_filename_png} (300 DPI)")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

plt.show()