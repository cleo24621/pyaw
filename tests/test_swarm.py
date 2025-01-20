import unittest

import numpy as np

from pyaw.swarm import quaternion_multiply, rotate_vector_by_quaternion
from datetime import datetime, timedelta

class MyTestCase(unittest.TestCase):
    def test_quaternions_of_mag_hr1b(self):
        """根据“Swarm Level 1b Processor Algorithms.pdf”中的四元数附录编写"""
        self.assertTrue(np.allclose(rotate_vector_by_quaternion(np.array([0,0,1]),np.array([0.183, 0.183, 0, 0.966])),
                        np.array([-0.354, 0.354, 0.866]),atol=1e-3))

if __name__ == '__main__':
    unittest.main()
