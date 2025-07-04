import numpy as np


class NEC2SCandSC2NEC:
    """Standard coordinate system transformation between nec and s/c (vector, not point)."""

    @staticmethod
    def get_rotmat_nec2sc_sc2nec(
        vsat_n: np.ndarray, vsat_e: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get rotation matrices to transform vector's coordinates between NEC and S/C coordinate systems.

        Args:
            vsat_n: velocity of satellite in the north direction.
            vsat_e: velocity of satellite in the east direction.

        Returns:
            rotation_matrix_2d_nec2sc, rotation_matrix_2d_sc2nec.
        """
        theta = np.arctan(vsat_e / vsat_n)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # Stack components to construct the rotation matrices
        rotation_matrix = np.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]])

        # Transpose axes to create a (3, 2, 2) array
        rotation_matrix_2d_nec2sc = np.transpose(rotation_matrix, (2, 0, 1))
        rotation_matrix_2d_sc2nec = rotation_matrix_2d_nec2sc.transpose(0, 2, 1)
        return rotation_matrix_2d_nec2sc, rotation_matrix_2d_sc2nec

    @staticmethod
    def do_rotation(
        coordinates1: np.ndarray,
        coordinates2: np.ndarray,
        rotation_matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Do the rotation of vector's coordinates between NEC and S/C coordinate systems.

        Args:
            coordinates1: N of NEC or X of S/C.
            coordinates2: E of NEC or Y of S/C.
            rotation_matrix: one of the rotation matrices returned by get_rotation_matrices_nec2sc_sc2nec.

        Returns:
            The transformed vector's coordinates.
        """
        vectors12 = np.stack((coordinates1, coordinates2), axis=1)
        vectors12_rotated = np.einsum("nij,nj->ni", rotation_matrix, vectors12)
        return vectors12_rotated[:, 0], vectors12_rotated[:, 1]
