import unittest

import numpy as np

from datetime import datetime, timedelta

from pyaw.core.process_data import SwarmProcess

import core.swarm


class MyTestCase(unittest.TestCase):
    def test_rotate_vector_by_quaternion(self):
        """根据“Swarm Level 1b Processor Algorithms.pdf”中的四元数附录编写"""
        rotate_vector_by_quaternion = core.swarm.NEC2SCofMAG.rotate_vector_by_quaternion
        self.assertTrue(
            np.allclose(
                rotate_vector_by_quaternion(
                    np.array([0, 0, 1]), np.array([0.183, 0.183, 0, 0.966])
                ),
                np.array([-0.354, 0.354, 0.866]),
                atol=1e-3,
            )
        )


if __name__ == "__main__":
    unittest.main()
