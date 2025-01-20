import unittest

import numpy as np
import pandas as pd

from pyaw.dmsp import SPDF

class TestDMSP(unittest.TestCase):
    def test_r_s3(self):
        fp = "D:\cleo\master\pyaw\data\DMSP\ssies3\dmsp-f18_ssies-3_thermal-plasma_202301010102_v01.cdf"
        dmsp = SPDF()
        result = dmsp.r_s3(fp)
        self.assertIsInstance(result, pd.DataFrame,"The result should be a DataFrame")
        self.assertGreater(len(result), 1, "The length of the DataFrame should be greater than 1")

    def test_r_ssm(self):
        """todo:: modify"""
        fp = "D:\cleo\master\pyaw\data\DMSP\ssies3\dmsp-f18_ssies-3_thermal-plasma_202301010102_v01.cdf"
        dmsp = SPDF()
        result = dmsp.r_s3(fp)
        self.assertIsInstance(result, pd.DataFrame,"The result should be a DataFrame")
        self.assertGreater(len(result), 1, "The length of the DataFrame should be greater than 1")

    def test_ssm_pre(self):
        df = pd.DataFrame(data=[1,2,3],index=pd.to_datetime([1490195805001000000,1490195805002000000,1490195807001000000], unit='ns'))
        df_re = df.resample('1s').mean()
        bool_ = df_re.equals(pd.DataFrame(data=[(1+2)/2,np.nan,3.0],index=pd.to_datetime([1490195805,1490195806,1490195807], unit='s')))
        self.assertTrue(bool_)

    def test_along_across_enu(self):
        fp_s3 = r"\\Diskstation1\file_three\Alfven wave\DMSP\spdf\f18\ssies3\2014\dmsp-f18_ssies-3_thermal-plasma_201401010124_v01.cdf"
        fp_ssm = r"\\Diskstation1\file_three\Alfven wave\DMSP\spdf\f18\ssm\2014\dmsp-f18_ssm_magnetometer_20140101_v1.0.4.cdf"
        dmspspdf = SPDF()
        s3_df = dmspspdf.r_s3(fp_s3)
        s3_df_pre = dmspspdf.s3_pre(s3_df)
        ssm_df = dmspspdf.r_ssm(fp_ssm)
        ssm_df_pre = dmspspdf.ssm_pre(ssm_df)
        clipped_ssm_df = dmspspdf.clip_ssm_by_ssies3(s3_df_pre, ssm_df_pre)
        # b
        com1, com2, com3 = dmspspdf.ssm_sc2s3_sc(clipped_ssm_df['b1_ssm_sc1'], clipped_ssm_df['b1_ssm_sc2'],
                                                 clipped_ssm_df['b1_ssm_sc3'])
        b1_enu1, b1_enu2, b1_enu3 = dmspspdf.s3_sc2enu(com1, com2, com3, clipped_ssm_df[
            ['sc_along_geo1', 'sc_along_geo2', 'sc_along_geo3', 'sc_across_geo1', 'sc_across_geo2', 'sc_across_geo3']])
        self.assertTrue(((b1_enu1 - clipped_ssm_df['b1_enu1']).abs().dropna() < 1e-4).all())  # the precession can be modified
        self.assertTrue(((b1_enu2 - clipped_ssm_df['b1_enu2']).abs().dropna() < 1e-4).all())
        self.assertTrue(((b1_enu3 - clipped_ssm_df['b1_enu3']).abs().dropna() < 1e-4).all())
if __name__ == '__main__':
    unittest.main()
