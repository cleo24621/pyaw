import unittest

import pandas as pd
import utils
from utils.orbit import (
    orbit_hemisphere_projection,
    orbit_hemispheres_projection,
    orbits_hemisphere_projection,
    orbits_hemispheres_projection,
)


class MyTestCase(unittest.TestCase):
    def test_orbit_hemisphere_projection(self):
        import os

        import pandas as pd

        from src.pyaw import ProjectConfigs

        satellite = "Swarm"
        data_dir_path = ProjectConfigs.data_dir_path
        file_name = "../data/Swarm/aux_/aux_SW_OPER_MAGA_LR_1B_12728_20160301T012924_20160301T030258.pkl"  # modify: different file
        file_path = os.path.join(data_dir_path, file_name)
        df = pd.read_pickle(file_path)
        lats = df["Latitude"].values
        lons = df["Longitude"].values
        indices = utils.orbit.get_nor_sou_split_indices_swarm_dmsp(lats)
        hemisphere_slice = slice(*indices[0])
        proj = "NorthPolarStereo"
        hemisphere_lats = lats[hemisphere_slice]
        hemisphere_lons = lons[hemisphere_slice]
        # success call
        fig, ax = orbit_hemisphere_projection(
            hemisphere_lons, hemisphere_lats, satellite=satellite, proj_method=proj
        )

    def test_orbit_hemispheres_projection(self):
        import os

        import pandas as pd

        from src.pyaw import ProjectConfigs

        # 主程序部分
        satellite = "Swarm"
        spacecraft = "A"
        orbit_number = 12727
        st = "20160229T235551"
        et = "20160301T012924"
        data_dir_path = ProjectConfigs.data_dir_path
        file_name = "../data/Swarm/aux_/aux_SW_OPER_MAGA_LR_1B_12728_20160301T012924_20160301T030258.pkl"  # modify: different file
        file_path = os.path.join(data_dir_path, file_name)
        df = pd.read_pickle(file_path)
        lats = df["Latitude"].values
        lons = df["Longitude"].values
        indices = utils.orbit.get_nor_sou_split_indices_swarm_dmsp(lats)
        northern_slice = slice(*indices[0])
        southern_slice = slice(*indices[1])
        lons_north = lons[northern_slice]
        lats_north = lats[northern_slice]

        lons_south = lons[southern_slice]
        lats_south = lats[southern_slice]

        # success call
        orbit_hemispheres_projection(
            lons_north, lats_north, lons_south, lats_south, satellite
        )

    def test_orbits_hemisphere_projection(self):
        import os.path

        from src.pyaw import ProjectConfigs

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
        lons_list = []
        lats_list = []
        for file_path in file_paths:
            df = pd.read_pickle(str(file_path))
            lats = df["Latitude"].values
            lons = df["Longitude"].values
            indices = utils.orbit.get_nor_sou_split_indices_swarm_dmsp(lats)
            nor_slice = slice(*indices[0])
            lons_list.append(lons[nor_slice])
            lats_list.append(lats[nor_slice])
        # success call
        orbits_hemisphere_projection(
            lons_list, lats_list, proj_method="NorthPolarStereo"
        )

    def test_orbits_hemispheres_projection(self):
        import os.path

        from src.pyaw import ProjectConfigs

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
            indices = utils.orbit.get_nor_sou_split_indices_swarm_dmsp(lats)
            nor_slice = slice(*indices[0])
            sou_slice = slice(*indices[1])
            lons_nor_list.append(lons[nor_slice])
            lats_nor_list.append(lats[nor_slice])
            lons_sou_list.append(lons[sou_slice])
            lats_sou_list.append(lats[sou_slice])
        # success call
        orbits_hemispheres_projection(
            lons_nor_list, lats_nor_list, lons_sou_list, lats_sou_list
        )


if __name__ == "__main__":
    unittest.main()
