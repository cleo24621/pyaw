import os
import unittest

import pandas as pd
from pandas import DataFrame
from utils.file import find_parent_directory

from core import zh1
from src.pyaw import Zh1Configs


class TestFGM(unittest.TestCase):
    def test_get_df(self):
        data_dir_path = find_parent_directory(os.path.dirname(__file__), "data")
        file_name = "../data/ZH1/CSES_01_HPM_5_L02_A2_096791_20191101_001933_20191101_005555_000.h5"
        file_path = os.path.join(data_dir_path,file_name)
        fgm = zh1.FGM(file_path)
        df = fgm.df
        # test
        self.assertIsInstance(df,DataFrame)
        self.assertTrue(not df.empty)
        for var in Zh1Configs.fgm_vars:
            self.assertTrue(var in df.columns, f"{var} is not in the dataframe")
        self.assertIsInstance(df.index,pd.RangeIndex)  # 检验索引列


class TestSCM(unittest.TestCase):
    def test_success_call(self):
        data_dir_path = find_parent_directory(os.path.dirname(__file__), "data")
        file_name = "../data/ZH1/CSES_01_SCM_1_L02_A2_096790_20191031_233256_20191101_000821_000.h5"
        file_path = os.path.join(data_dir_path,file_name)
        scm = zh1.SCM(file_path)  # 成功调用类
        for var in Zh1Configs.scm_ulf_vars:
            self.assertTrue(var in scm.dfs.keys(), f"the DataFrame of {var} is not in the dataframes")


# todo: resample之后
# def test_get_signal(self):
#     for var in configs.scm_ulf_resample_vars:
#         signal = scm._get_resample_signal(var)
#         self.assertTrue(type(signal) == pd.Series)
#         self.assertTrue(not signal.empty)
#         self.assertTrue(signal.index.dtype == 'datetime64[ns]')
#         self.assertTrue(signal.index.freq.name == 'us' and signal.index.freq.n == 62500)
#         print(signal.name, ':')
#         print(signal.head())
#         print('------')
#
#
class TestEFD(unittest.TestCase):
    def test_success_call(self):
        data_dir_path = find_parent_directory(os.path.dirname(__file__), "data")
        file_name = "../data/ZH1/CSES_01_EFD_1_L2A_A1_175380_20210401_003440_20210401_010914_000.h5"
        file_path = os.path.join(data_dir_path,file_name)
        efd = zh1.EFD(file_path)  # 成功调用类
        for var in Zh1Configs.efd_ulf_vars:
            self.assertTrue(var in efd.dfs.keys(), f"the DataFrame of {var} is not in the dataframes")

# todo: resample之后
# def test_get_signal(self):
# for var in configs.efd_ulf_resample_vars:
#     signal = self.efd._get_resample_signal(var)
#     self.assertTrue(type(signal) == pd.Series)
#     self.assertTrue(not signal.empty)
#     self.assertTrue(signal.index.dtype == 'datetime64[ns]')
#     self.assertTrue(signal.index.freq.name == 'ms' and signal.index.freq.n == 40)
#     print(signal.name, ':')
#     print(signal.head())
#     print('------')


class TestEFDSCMClip(unittest.TestCase):
    def test_success_call(self):
        data_dir_path = find_parent_directory(os.path.dirname(__file__), "data")
        # same orbit number, same descend
        scm_file_name = "../data/ZH1/CSES_01_SCM_1_L02_A2_175380_20210401_003346_20210401_010912_000.h5"
        efd_file_name = "../data/ZH1/CSES_01_EFD_1_L2A_A1_175380_20210401_003440_20210401_010914_000.h5"
        scm_file_path = os.path.join(data_dir_path,scm_file_name)
        efd_file_path = os.path.join(data_dir_path,efd_file_name)
        start_time = pd.Timestamp(year=2021,month=4,day=1,hour=0,minute=35)
        end_time = pd.Timestamp(year=2021,month=4,day=1,hour=1,minute=0)
        scm_efd = zh1.SCMEFDUlf(st=start_time,et=end_time,fp_scm=scm_file_path,fp_efd=efd_file_path)


if __name__ == '__main__':
    unittest.main()
