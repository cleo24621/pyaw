import unittest

from utils import convert_tstr2dt64, get_3arrs, get_rotmat_nec2sc_sc2nec, do_rotation
import numpy as np


class TestUtils(unittest.TestCase):
    def test_convert_tstr2dt64(self):
        tstr = "20160311T064700"
        dt64 = convert_tstr2dt64(tstr)
        self.assertIsInstance(dt64, np.datetime64, "The result should be a numpy datetime64 object")
        self.assertTrue(dt64.dtype == np.dtype('datetime64[ns]'), "The dtype should be datetime64[ns]")

    def test_get_3arrs(self):
        arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
        arr1,arr2,arr3 = get_3arrs(arr)
        self.assertTrue(np.array_equal(arr1,np.array([1,4,7])))
        self.assertTrue(np.array_equal(arr2,np.array([2,5,8])))
        self.assertTrue(np.array_equal(arr3,np.array([3,6,9])))

    def test_rotation1(self):
        VsatN = np.array([1,2,3])
        VsatE = np.array([4,5,6])
        rotation_matrix_2d_nec2sc, rotation_matrix_2d_sc2nec = get_rotmat_nec2sc_sc2nec(VsatN,VsatE)
        self.assertEqual(len(rotation_matrix_2d_nec2sc.shape),3)
        self.assertEqual(len(rotation_matrix_2d_sc2nec.shape),3)
        Vsatx,Vsaty = do_rotation(VsatN,VsatE,rotation_matrix_2d_nec2sc)
        self.assertTrue(np.allclose(Vsatx,np.sqrt(VsatN**2+VsatE**2)))
        self.assertTrue(np.allclose(Vsaty,0))

    # def test_rotation2(self):
    #     """
    #     通过对比vires输出的模型磁场和tct的观测磁场转换到NEC坐标系，验证自己对于tct数据产品中的x,y,z坐标系理解是否正确。
    #     """
    #     df_igrf_vires = pd.read_pickle("../data/result/SW_EXPT_EFIA_TCT16_IGRF_20160311T064700_20160311T064701_vires.pkl")
    #     Bn_vires,Be_vires,Bc_vires = get_3arrs(df_igrf_vires['B_NEC_IGRF'].values)
    #
    #     df_m = pd.read_pickle("../data/EFIA_TCT16_20160311T064640_20160311T064920.pkl")
    #     df_m = df_m[(df_m.index> pd.Timestamp("2016-03-11 06:47:00")) & (df_m.index< pd.Timestamp("2016-03-11 06:47:01"))]
    #     df_m = df_m[['VsatN','VsatE','Bx','By']]
    #     self.assertTrue(np.array_equal(df_m.index.values,df_igrf_vires.index.values))
    #     rotation_matrix_2d_nec2sc, rotation_matrix_2d_sc2nec = get_rotmat_nec2sc_sc2nec(df_m['VsatN'].values, df_m['VsatE'].values)
    #     Bn,Be = do_rotation(-df_m['Bx'].values,-df_m['By'].values,rotation_matrix_2d_sc2nec)
    #     self.assertTrue(np.allclose(Bn,Bn_vires,rtol=1e-2))  # 注意这里是观测磁场和模型磁场的比较！！！
    #     # 我对于tct数据产品中的x,y,z坐标系的认识是正确的




if __name__ == '__main__':
    unittest.main()
