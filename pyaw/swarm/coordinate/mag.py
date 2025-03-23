import numpy as np
from nptyping import NDArray, Shape, Float64


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
            variable_nec.dtype == object and variable_quaternion_nec_crf.dtype == object
        )
        self.variable_nec = variable_nec
        self.variable_quaternion_nec_crf = variable_quaternion_nec_crf
        self.variable_sc = self._calculate_rotated_vectors()

    def _quaternion_multiply(
        self, p: NDArray[Shape["4"], Float64], q: NDArray[Shape["4"], Float64]
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

    def _rotate_vector_by_quaternion(
        self,
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

        vector_quat = np.array([*vector, 0])  # Convert vector to quaternion form
        quaternion_conj = np.array(
            [-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]]
        )
        vector_rotated_quat = self._quaternion_multiply(
            self._quaternion_multiply(quaternion_conj, vector_quat), quaternion
        )
        return vector_rotated_quat[:3]  # Return only the vector part

    def _calculate_rotated_vectors(
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
            variable_sc[i] = self._rotate_vector_by_quaternion(
                vector, quaternion_crf_nec
            )
        assert variable_sc.dtype == object
        return variable_sc
