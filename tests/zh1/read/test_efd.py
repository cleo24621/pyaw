import unittest
import os

from pyaw.zh1.read.efd import EFDUlf


class TestSCMUlf(unittest.TestCase):
    def test_success_call(self):
        file_name = "CSES_01_EFD_1_L2A_A1_175380_20210401_003440_20210401_010914_000.h5"
        current_dir = os.path.dirname(os.path.abspath(__file__))
        grandparent_dir = os.path.abspath(os.path.join(current_dir,"../../"))
        file_path = os.path.join(grandparent_dir,f"data/{file_name}")
        EFDUlf(file_path)  # 成功调用类


if __name__ == '__main__':
    unittest.main()
