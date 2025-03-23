import unittest
import os

from pyaw.zh1.read.scm import SCMUlf


class TestSCMUlf(unittest.TestCase):
    def test_success_call(self):
        file_name = "CSES_01_SCM_1_L02_A2_096790_20191031_233256_20191101_000821_000.h5"
        current_dir = os.path.dirname(os.path.abspath(__file__))
        grandparent_dir = os.path.abspath(os.path.join(current_dir,"../../"))
        file_path = os.path.join(grandparent_dir,f"data/{file_name}")
        SCMUlf(file_path)  # 成功调用类


if __name__ == '__main__':
    unittest.main()
