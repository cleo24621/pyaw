# -*- coding: utf-8 -*-
"""
@Author      : 13927
@Date        : 3/21/2025 9:55
@Project     : pyaw
@Description : 描述此文件的功能和用途。

@Copyright   : Copyright (c) 2025, 13927
@License     : 使用的许可证（如 MIT, GPL）
@Last Modified By : 13927
"""

import numpy as np
from nptyping import NDArray, Shape, Float64
from typing import Optional,Any


class BaseProcess:
    """
    一些重复的方法
    """

    @staticmethod
    def get_split_indices(array: NDArray,indicator:Optional[str]=None):
        """
        get the indices that split array into northern and southern.
        todo: 也许需要添加处理纬度等于0时属于北半球或南半球的情况（会有吗？（极端情况：0.00001））
        Args:
            indicator: 对于swarm,dmsp而言，indicator=None，即调用该方法时不需要输入indicator；对于zh1而言，indicator="0" or indicator="1".
            array: latitudes. pattern is [south,north], or [south,north,south(few)]

        """
        neg_indices = np.where(array < 0)[0]
        if not neg_indices.size:
            return (0, len(array)), (len(array), len(array))

        start_south = neg_indices[0]
        pos_indices = np.where(array[start_south:] >= 0)[0]
        end_south = start_south + pos_indices[0] if pos_indices.size else len(array)

        return (0, start_south), (start_south, end_south)


class SwarmProcess(BaseProcess):
    class MagPreprocess:
        """
        有关MAG产品的预处理
        """

        class NEC2SC:
            """
            Quaternion using in swarm 1b product algorithm. it is a little different from the general definition of quaternion.
            tct16产品没有'q_NEC_CRF'变量.
            """

            def __init__(
                self,
                variable_nec: NDArray,
                variable_quaternion_nec_crf: NDArray,
            ) -> None:
                """

                Args:
                    variable_nec: 外层数组，形状 (N,)，每个元素是形状 (3,) 的浮点数数组. such as np.array([np.array([1,1,1]),np.array([2,2,2]),...])
                    variable_quaternion_nec_crf: ~ (Quaternion, transformation: NEC ← CRF)
                """
                assert (
                    variable_nec.dtype == object
                    and variable_quaternion_nec_crf.dtype == object
                )
                self.variable_nec = variable_nec
                self.variable_quaternion_nec_crf = variable_quaternion_nec_crf

            @staticmethod
            def rotate_vector_by_quaternion(
                vector: NDArray[Shape["3"], Float64],
                quaternion: NDArray[Shape["4"], Float64],
            ) -> NDArray[Shape["3"], Float64]:
                """
                Rotate vector using quaternion. Refer to the docs of 'quaternion_multiply()'.

                Args:
                    vector: the vector needed to be rotated, (np.array([v1,v2,v3]))
                    quaternion: the corresponded quaternion, (np.array([q1,q2,q3,q4]))

                Returns:
                    the rotated vector
                """

                def quaternion_multiply(
                    p: NDArray[Shape["4"], Float64], q: NDArray[Shape["4"], Float64]
                ) -> NDArray[Shape["4"], Float64]:
                    """
                    Define quaternion multiplication (differ from the general multiplication).
                    quaternion using in swarm 1b product algorithm. it is a little different from the general definition of quaternion.

                    Args:
                        p: one of the multiplication vector (np.array([p1,p2,p3,p4]))
                        q: another of the multiplication vector (np.array([q1,q2,q3,q4]))

                    Returns:
                        NDArray[np.float64, (4,)]: the multiplication result
                    """
                    p1, p2, p3, p4 = p
                    q1, q2, q3, q4 = q
                    return np.array(
                        [
                            p1 * q4 + p2 * q3 - p3 * q2 + p4 * q1,
                            -p1 * q3 + p2 * q4 + p3 * q1 + p4 * q2,
                            p1 * q2 - p2 * q1 + p3 * q4 + p4 * q3,
                            -p1 * q1 - p2 * q2 - p3 * q3 + p4 * q4,
                        ]
                    )

                vector_quat = np.array(
                    [*vector, 0]
                )  # Convert vector to quaternion form
                quaternion_conj = np.array(
                    [-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]]
                )
                vector_rotated_quat = quaternion_multiply(
                    quaternion_multiply(quaternion_conj, vector_quat), quaternion
                )
                return vector_rotated_quat[:3]  # Return only the vector part

            def calculate_rotated_vectors(
                self,
            ) -> NDArray:
                """
                Get the variable data in SC Frame. (Note that the SC Frame is different between Swarm and DMSP)
                Args:

                Returns:
                    the variable data in SC Frame
                """
                n = len(self.variable_nec)
                variable_sc = np.empty((n, 3))  # 预分配数组
                for i, (vector, q) in enumerate(
                    zip(self.variable_nec, self.variable_quaternion_nec_crf)
                ):
                    quaternion_crf_nec = np.array(
                        [-q[0], -q[1], -q[2], q[3]]
                    )  # Quaternion, transformation: CRF ← NEC
                    variable_sc[i] = self.rotate_vector_by_quaternion(
                        vector, quaternion_crf_nec
                    )
                assert variable_sc.dtype == object
                return variable_sc

    class Hemisphere(BaseProcess):
        pass

class DmspProcess(BaseProcess):
    class SpdfRead:
        pass

    class MadrigalRead:
        pass

    class Hemisphere(BaseProcess):
        pass


class Zh1Process(BaseProcess):
    class Hemisphere(BaseProcess):
        @staticmethod
        def get_orbit_num_indicator_st_et(file_name):
            """
            因为不同2级产品的命名格式是固定的，所以当前方法适用于所有2a级产品的文件名（参考文档）。
            1: ascending (south to north)
            0: descending (north to south)
            Args:
                file_name: now only support the efd 2a data filename
            """
            parts = file_name.split("_")
            part = parts[6]
            assert part[-1] in ["0", "1"]
            start_time = parts[7] + "_" + parts[8]
            end_time = parts[9] + "_" + parts[10]
            return parts[6][:-1], parts[6][-1], start_time, end_time

        @staticmethod
        def get_split_indices(array: NDArray,indicator:Optional[str]=None):
            assert indicator in ["1", "0"]
            if all(array > 0):
                return (0, len(array)), (len(array), len(array))
            elif all(array < 0):
                return (len(array), len(array)), (0, len(array))
            elif indicator == "1":
                start_north = np.where(array > 0)[0][0]
                return (start_north, None), (0, start_north)  # north, south slice
            else:
                start_south = np.where(array < 0)[0][0]
                return (0, start_south), (start_south, None)  # north, south slice


def main():
    pass


if __name__ == "__main__":
    main()