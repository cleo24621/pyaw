import unittest

import numpy as np

from pyaw.utils import coordinate,orbit

class MyTestCase(unittest.TestCase):
    def test_orbit_hemisphere_with_vector_projection(self):
        import os

        import pandas as pd
        from matplotlib import pyplot as plt

        from pyaw.configs import ProjectConfigs
        from pyaw.utils import orbit

        satellite = "Swarm"
        data_dir_path = ProjectConfigs.data_dir_path
        file_name = "SW_EXPT_EFIA_TCT16_12728_20160301T012924_20160301T030258.pkladdddaaaaaaaaaaaaad"  # modify: different file
        file_path = os.path.join(data_dir_path, file_name)
        df = pd.read_pickle(file_path)
        lats = df["Latitude"].values
        lons = df["Longitude"].values
        indices = orbit.get_nor_sou_split_indices_swarm_dmsp(lats)
        hemisphere_slice = slice(*indices[0])
        proj_method = "NorthPolarStereo"
        hemisphere_lats = lats[hemisphere_slice]
        hemisphere_lons = lons[hemisphere_slice]

        Ehx = df['Ehx'].values
        Ehy = df['Ehy'].values
        VsatN = df['VsatN'].values
        VsatE = df['VsatE'].values

        # 计算旋转矩阵
        rotmat_nec2sc, rotmat_sc2nec = coordinate.NEC2SCandSC2NEC.get_rotmat_nec2sc_sc2nec(VsatN, VsatE)
        E_north, E_east = coordinate.NEC2SCandSC2NEC.do_rotation(-Ehx, -Ehy, rotmat_sc2nec)

        # 正则化矢量
        magnitudes = np.sqrt(E_east ** 2 + E_north ** 2)
        min_mag = 1e-8
        max_mag = np.max(np.clip(magnitudes, a_min=min_mag, a_max=None))
        scale_factor = 1.8
        E_east_norm = (E_east / max_mag) * scale_factor
        E_north_norm = (E_north / max_mag) * scale_factor

        orbit.orbit_hemisphere_with_vector_projection(lons=lons,lats=lats,vector_east=E_east_norm,vector_north=E_north_norm,proj_method=proj_method)
        plt.show()




if __name__ == '__main__':
    unittest.main()
