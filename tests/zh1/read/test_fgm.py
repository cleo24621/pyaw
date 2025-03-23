import unittest

import pandas as pd


class TestFGM(unittest.TestCase):
    def test_get_df(self):
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录
        grandparent_dir = os.path.abspath(os.path.join(current_dir, "../../"))  # 上上级目录
        file_path = os.path.join(grandparent_dir, "data", "CSES_01_HPM_5_L02_A2_096791_20191101_001933_20191101_005555_000.h5")  # 组合路径

        from pyaw.zh1.read import fgm
        from pandas import DataFrame
        df = fgm.get_df(file_path)
        self.assertIsInstance(df,DataFrame)  # 检验返回类型
        self.assertIsInstance(df.index,pd.RangeIndex)  # 检验索引列


if __name__ == '__main__':
    unittest.main()
