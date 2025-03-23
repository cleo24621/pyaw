import unittest
from pathlib import Path

import pandas as pd


class TestBaseProcess(unittest.TestCase):
    def test_get_split_indices(self):
        from pyaw.core.process_data import BaseProcess

        # 构建测试数据绝对路径
        current_dir = Path(__file__).parent
        test_data_path = (
            current_dir.parent
            / "data"
            / "only_gdcoors_SW_OPER_MAGA_LR_1B_12727_20160229T235551_20160301T012924.pkl"
        )
        # 加载测试数据（确保文件读取后关闭，释放内存）
        with open(test_data_path, "rb") as f:
            test_df = pd.read_pickle(f)
        latitudes = test_df["Latitude"].values
        # 执行被测试方法
        indices = BaseProcess.get_split_indices(latitudes)
        # 验证返回结构
        self.assertIsInstance(indices, tuple, "The type of slices should be tuple")
        self.assertEqual(len(indices), 2, "Should contain two slices")
        self.assertIsInstance(indices[0], tuple, "The slice of slices should be tuple")
        self.assertIsInstance(indices[1], tuple, "The slice of slices should be tuple")
        # 解包切片并验证有效性
        northern_slice = slice(*indices[0])
        southern_slice = slice(*indices[1])
        # 验证纬度数据分割正确性
        northern_lats = latitudes[northern_slice]
        southern_lats = latitudes[southern_slice]
        # 验证纬度方向（仅在存在数据时验证）（允许不存在数据）
        if len(northern_lats) > 0:
            self.assertTrue(
                (northern_lats > 0).all(),
                "All northern latitudes should be positive when present",
            )
        if len(southern_lats) > 0:
            self.assertTrue(
                (southern_lats < 0).all(),
                "All southern latitudes should be negative when present",
            )


class TestZh1ProcessHemisphere(unittest.TestCase):
    def test_get_orbit_num_indicator_st_et(self):
        from pyaw.core.process_data import Zh1Process

        # 不同2a产品的文件名
        file_names = {
            "fgm": "CSES_01_HPM_5_L02_A2_096791_20191101_001933_20191101_005555_000.h5",
            "efd": "CSES_01_EFD_1_L2A_A1_175371_20210331_234716_20210401_002158_000.h5",
            "scm": "CSES_01_SCM_1_L02_A2_096790_20191031_233256_20191101_000821_000.h5",
            "lap": "CSES_01_LAP_1_L02_A3_096790_20191031_233232_20191101_000948_000.h5",
        }

        # 函数名
        get_orbit_num_indicator_st_et = (
            Zh1Process.Hemisphere.get_orbit_num_indicator_st_et
        )

        # 验证
        for file_name in file_names.values():
            orbit_number, indicator, start_time, end_time = (
                Zh1Process.Hemisphere.get_orbit_num_indicator_st_et(file_name)
            )
            self.assertTrue(len(orbit_number) == 5, "轨道号（5 位数字）")
            self.assertTrue(
                len(start_time) == 15,
                "数据起始时间，采用 14 位数字表示（包含分隔符'_'，所以长度因为15）",
            )
            self.assertTrue(
                len(start_time) == len(end_time), "数据结束时间，采用 14 位数字表示"
            )

    def test_get_split_indices(self):
        from pyaw.core.process_data import Zh1Process
        from pyaw.core.read_data import Zh1Read

        get_orbit_num_indicator_st_et = (
            Zh1Process.Hemisphere.get_orbit_num_indicator_st_et
        )
        get_split_indices = Zh1Process.Hemisphere.get_split_indices

        # 0 descending (north to south)

        # path
        file_name = "CSES_01_EFD_1_L2A_A1_175380_20210401_003440_20210401_010914_000.h5"
        current_dir = Path(__file__).parent
        test_data_path = current_dir.parent / "data" / file_name

        # test
        indicator = get_orbit_num_indicator_st_et(file_name)[1]
        efd = Zh1Read.EFDUlf(test_data_path)
        dfs = efd.dfs
        lats = dfs["GEO_LAT"].squeeze().values
        indices = get_split_indices(lats, indicator)
        northern_slice = slice(*indices[0])
        southern_slice = slice(*indices[1])
        orbit_lats_north = lats[northern_slice]
        orbit_lats_south = lats[southern_slice]
        self.assertTrue(all(orbit_lats_north > 0))
        self.assertTrue(all(orbit_lats_south < 0))

        # 1 ascending (south to north)

        # path
        file_name = "CSES_01_EFD_1_L2A_A1_175381_20210401_012158_20210401_015642_000.h5"
        current_dir = Path(__file__).parent
        test_data_path = current_dir.parent / "data" / file_name
        indicator = get_orbit_num_indicator_st_et(file_name)[1]
        efd = Zh1Read.EFDUlf(test_data_path)
        dfs = efd.dfs
        lats = dfs["GEO_LAT"].squeeze().values
        indices = get_split_indices(lats, indicator)
        northern_slice = slice(*indices[0])
        southern_slice = slice(*indices[1])
        orbit_lats_north = lats[northern_slice]
        orbit_lats_south = lats[southern_slice]
        self.assertTrue(all(orbit_lats_north > 0))
        self.assertTrue(all(orbit_lats_south < 0))


if __name__ == "__main__":
    unittest.main()
