import unittest

import pandas as pd

import pyaw.zh1
from pyaw import configs


class TestFGM(unittest.TestCase):
    def test_get_df(self):
        fp = r"\\Diskstation1\file_three\aw\zh1\hpm\fgm\201911\CSES_01_HPM_5_L02_A2_096791_20191101_001933_20191101_005555_000.h5"
        fgm = pyaw.zh1.FGM(fp)
        df = fgm.get_df()
        print(type(df))
        self.assertTrue(type(df) == pd.DataFrame)
        self.assertTrue(not df.empty)
        print(df.info(), '\n')
        print(df.columns)
        for var in configs.fgm_vars:
            self.assertTrue(var in df.columns, f"{var} is not in the dataframe")


class TestSCMULF(unittest.TestCase):
    def test_get_signal(self):
        fp = r"\\Diskstation1\file_three\aw\zh1\scm\ulf\201911\CSES_01_SCM_1_L02_A2_096790_20191031_233256_20191101_000821_000.h5"
        scm = pyaw.zh1.SCMULF(fp)
        for var in configs.scm_ulf_vars:
            self.assertTrue(var in scm.dfs.keys(), f"the DataFrame of {var} is not in the dataframes")
        # for var in configs.scm_ulf_resample_vars:
        #     signal = scm._get_resample_signal(var)
        #     self.assertTrue(type(signal) == pd.Series)
        #     self.assertTrue(not signal.empty)
        #     self.assertTrue(signal.index.dtype == 'datetime64[ns]')
        #     self.assertTrue(signal.index.freq.name == 'us' and signal.index.freq.n == 62500)
        #     print(signal.name, ':')
        #     print(signal.head())
        #     print('------')


class TestEFDULF(unittest.TestCase):
    fp = r"\\Diskstation1\file_three\aw\zh1\efd\ulf\201911\CSES_01_EFD_1_L02_A1_096790_20191031_233350_20191101_000824_000.h5"
    efd = pyaw.zh1.EFDULF(fp)

    def test_init(self):
        self.assertEqual(self.efd.fs, 125)
        # self.assertEqual(self.efd.target_fs, 25)

    def test_get_signal(self):
        for var in configs.efd_ulf_vars:
            self.assertTrue(var in self.efd.dfs.keys(), f"the DataFrame of {var} is not in the dataframes")
        # for var in configs.efd_ulf_resample_vars:
        #     signal = self.efd._get_resample_signal(var)
        #     self.assertTrue(type(signal) == pd.Series)
        #     self.assertTrue(not signal.empty)
        #     self.assertTrue(signal.index.dtype == 'datetime64[ns]')
        #     self.assertTrue(signal.index.freq.name == 'ms' and signal.index.freq.n == 40)
        #     print(signal.name, ':')
        #     print(signal.head())
        #     print('------')


class TestOther(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
