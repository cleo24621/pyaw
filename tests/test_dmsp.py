import os.path
import unittest

import numpy as np
import pandas as pd
from pandas import DataFrame

from pyaw.configs import ProjectConfigs
from pyaw.core.dmsp import SPDF

data_dir_path = ProjectConfigs.data_dir_path


class TestSSIES3(unittest.TestCase):
    def test_ssies3(self):
        file_name = "dmsp-f18_ssies-3_thermal-plasma_202301010102_v01.cdf"
        file_path = os.path.join(data_dir_path, file_name)
        # 检验是否成功调用
        ssies3 = SPDF.SSIES3(file_path)
        # 检验返回类型
        self.assertIsInstance(
            ssies3.original_df, DataFrame, "The data structure should be a DataFrame"
        )
        # 检验数据非空
        self.assertTrue(
            not ssies3.original_df.empty, "The DataFrame shouldn't be emtpy."
        )

    def test_quality_process(self):
        # need or not
        pass


class TestSSM(unittest.TestCase):
    def test_ssm(self):
        file_name = "dmsp-f18_ssm_magnetometer_20140101_v1.0.4.cdf"
        file_path = os.path.join(data_dir_path, file_name)
        # 检验是否成功调用
        ssm = SPDF.SSM(file_path)
        # 检验返回类型
        self.assertIsInstance(
            ssm.original_df, DataFrame, "The data structure should be a DataFrame"
        )
        self.assertTrue(not ssm.original_df.empty, "The DataFrame shouldn't be emtpy.")

    def test_ssm_pre(self):
        """
        测试ssm._process()中的时间序列重采样符合设计预期
        Returns:

        """
        df = pd.DataFrame(
            data=[1, 2, 3],
            index=pd.to_datetime(
                [1490195805001000000, 1490195805002000000, 1490195807001000000],
                unit="ns",
            ),
        )
        df_re = df.resample("1s").mean()
        bool_ = df_re.equals(
            pd.DataFrame(
                data=[(1 + 2) / 2, np.nan, 3.0],
                index=pd.to_datetime([1490195805, 1490195806, 1490195807], unit="s"),
            )
        )
        self.assertTrue(bool_)

    # todo: clip之后
    # def test_along_across_enu(self):
    #     fp_s3 = r"\\Diskstation1\file_three\Alfven wave\DMSP\spdf\f18\ssies3\2014\dmsp-f18_ssies-3_thermal-plasma_201401010124_v01.cdf"
    #     fp_ssm = r"\\Diskstation1\file_three\Alfven wave\DMSP\spdf\f18\ssm\2014\dmsp-f18_ssm_magnetometer_20140101_v1.0.4.cdf"
    #     dmspspdf = SPDF()
    #     s3_df = dmspspdf.r_s3(fp_s3)
    #     s3_df_pre = dmspspdf._quality_process(s3_df)
    #     ssm_df = dmspspdf.r_ssm(fp_ssm)
    #     ssm_df_pre = dmspspdf.ssm_pre(ssm_df)
    #     clipped_ssm_df = dmspspdf.clip_ssm_by_ssies3(s3_df_pre, ssm_df_pre)
    #     # b
    #     com1, com2, com3 = dmspspdf.ssm_sc2s3_sc(clipped_ssm_df['b1_ssm_sc1'], clipped_ssm_df['b1_ssm_sc2'],
    #                                              clipped_ssm_df['b1_ssm_sc3'])
    #     b1_enu1, b1_enu2, b1_enu3 = dmspspdf.s3_sc2enu(com1, com2, com3, clipped_ssm_df[
    #         ['sc_along_geo1', 'sc_along_geo2', 'sc_along_geo3', 'sc_across_geo1', 'sc_across_geo2', 'sc_across_geo3']])
    #     self.assertTrue(((b1_enu1 - clipped_ssm_df['b1_enu1']).abs().dropna() < 1e-4).all())  # the precession can be modified
    #     self.assertTrue(((b1_enu2 - clipped_ssm_df['b1_enu2']).abs().dropna() < 1e-4).all())
    #     self.assertTrue(((b1_enu3 - clipped_ssm_df['b1_enu3']).abs().dropna() < 1e-4).all())


class TestSSIES3CoupleSSM(unittest.TestCase):
    def test_success_call(self):
        # not cross day, same day
        file_name_ssies3 = "dmsp-f18_ssies-3_thermal-plasma_201401010124_v01.cdf"
        file_name_ssm = "dmsp-f18_ssm_magnetometer_20140101_v1.0.4.cdf"
        file_path_ssies3 = os.path.join(data_dir_path, file_name_ssies3)
        file_path_ssm = os.path.join(data_dir_path, file_name_ssm)
        # 成功调用
        ssies3_couple_ssm = SPDF.SSIES3CoupleSSM(file_path_ssies3, file_path_ssm)

        # cross day, same day
        file_name_ssies3 = "dmsp-f18_ssies-3_thermal-plasma_201401012329_v01.cdf"
        file_path_ssies3 = os.path.join(data_dir_path, file_name_ssies3)
        # 调用时抛出正确错误
        with self.assertRaises(AssertionError) as cm:
            ssies3_couple_ssm = SPDF.SSIES3CoupleSSM(file_path_ssies3, file_path_ssm)
        self.assertEqual(str(cm.exception), "ssies3文件不跨天")

        # not cross day, not same day
        file_name_ssies3 = "dmsp-f18_ssies-3_thermal-plasma_202301010102_v01.cdf"
        file_path_ssies3 = os.path.join(data_dir_path, file_name_ssies3)
        # 调用时抛出正确错误
        with self.assertRaises(AssertionError) as cm:
            ssies3_couple_ssm = SPDF.SSIES3CoupleSSM(file_path_ssies3, file_path_ssm)
        self.assertEqual(str(cm.exception), "ssies3,ssm应为同一天")

    def test_clip(self):
        # ssies3 instance
        file_name = "dmsp-f18_ssies-3_thermal-plasma_202301010102_v01.cdf"
        file_path = os.path.join(data_dir_path, file_name)
        ssies3 = SPDF.SSIES3(file_path)

        # ssm instance
        file_name = "dmsp-f18_ssm_magnetometer_20140101_v1.0.4.cdf"
        file_path = os.path.join(data_dir_path, file_name)
        # 检验是否成功调用
        ssm = SPDF.SSM(file_path)


if __name__ == "__main__":
    unittest.main()
