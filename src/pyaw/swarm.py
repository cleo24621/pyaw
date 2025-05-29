import time
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from viresclient import SwarmRequest

from pyaw.satellite import NEC2SCandSC2NEC

SAMPLE_RATES = {"tct16": 16, "mag_hr": 50, "mag_lr": 1}

MAG_VARIABLES = ["q_NEC_CRF", "B_NEC"]

# viresclient SwarmRequest configurations
SWARM_REQUEST_CONFIG = {
    "auxiliaries": [
        "AscendingNodeLongitude",
        "QDLat",
        "QDLon",
        "QDBasis",
        "MLT",
        "SunDeclination",
    ],
    "models": ["IGRF"],
    "collections": {
        "EFI_TCT16": [
            "SW_EXPT_EFIA_TCT16",
            "SW_EXPT_EFIB_TCT16",
            "SW_EXPT_EFIC_TCT16",
        ],
        "EFI_TCT02": [
            "SW_EXPT_EFIA_TCT02",
            "SW_EXPT_EFIB_TCT02",
            "SW_EXPT_EFIC_TCT02",
        ],
        "MAG": ["SW_OPER_MAGA_LR_1B", "SW_OPER_MAGB_LR_1B", "SW_OPER_MAGC_LR_1B"],
        "MAG_HR": [
            "SW_OPER_MAGA_HR_1B",
            "SW_OPER_MAGB_HR_1B",
            "SW_OPER_MAGC_HR_1B",
        ],
    },
}


class TCT16:
    """
    Process tct16 l1b production.
    For tct16, use 'nan_outliers_by_std_dev()' to process outlier.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.dataframe = pd.read_pickle(file_path)
        self.VsatN = self.dataframe["VsatN"].values
        self.VsatE = self.dataframe["VsatE"].values

    def sc2nec(self, vectorx, vectory):
        """
        Vector is transformed from s/c coordinate system to nec coordinate system.
        x, y -> n, e.

        Args:
            vectorx: x component of a vector. For electric filed, need input -Ex not Ex?
            vectory: vectory and vectorx are similar.

        Returns:
            A tuple, the first is the north component, the second is the east component.
        """
        __, rotmat_sc2nec = NEC2SCandSC2NEC.get_rotmat_nec2sc_sc2nec(
            self.VsatN, self.VsatE
        )
        return NEC2SCandSC2NEC.do_rotation(vectorx, vectory, rotmat_sc2nec)


class MAG:
    """process mag (hr and lr) l1b production"""

    def __init__(self, file_path):
        self.file_path = file_path
        self.dataframe = pd.read_pickle(file_path)
        # Quaternion transform using in swarm mag 1b product algorithm.
        # It is a little different from the general definition of quaternion transform.
        self.variable_b_nec = self.dataframe["B_NEC"].values
        self.variable_quaternion_nec_crf = self.dataframe["q_NEC_CRF"].values
        self.b_sc = self._calculate_rotated_vectors()

    @staticmethod
    def _quaternion_multiply(p: NDArray, q: NDArray) -> NDArray:
        """
        Define quaternion multiplication (differ from the general multiplication).
        quaternion using in swarm 1b product algorithm.
        It is a little different from the general definition of quaternion.

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
        vector: NDArray,
        quaternion: NDArray,
    ) -> NDArray:
        """
        Rotate vector using the quaternion.

        Args:
            vector: the vector needed to be rotated, (np.array([v1,v2,v3]))
            quaternion: the corresponded quaternion, (np.array([q1,q2,q3,q4]))

        Returns:
            The rotated vector.
        """

        vector_quat = np.array([*vector, 0])  # Convert vector to quaternion form
        quaternion_conj = np.array(
            [-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]]
        )
        vector_rotated_quat = self._quaternion_multiply(
            self._quaternion_multiply(quaternion_conj, vector_quat), quaternion
        )
        return vector_rotated_quat[:3]  # Return only the vector part

    def _calculate_rotated_vectors(self) -> NDArray:
        """
        Get the variable data in s/c coordinate system.
        Note that the definition of s/c coordinate system is different between Swarm and DMSP.

        Args:

        Returns:
            The variable data in s/c coordinate system.
        """
        n = len(self.variable_b_nec)
        variable_sc = np.empty((n, 3))  # 预分配数组
        for i, (vector, q) in enumerate(
            zip(self.variable_b_nec, self.variable_quaternion_nec_crf)
        ):
            quaternion_crf_nec = np.array(
                [-q[0], -q[1], -q[2], q[3]]
            )  # Quaternion, transformation: CRF ← NEC
            variable_sc[i] = self._rotate_vector_by_quaternion(
                vector, quaternion_crf_nec
            )
        # assert variable_sc.dtype == object
        return variable_sc


def download_orbit_collection(
    satellite: str,
    collection: str,
    orbit_number: int,
    download_type: Union[str],
    sdir: Path,
) -> None:
    """
    By using the 'viresclient.SwarmRequest',
    download one orbit of collection data as a DataFrame and save it as a '.pkl' file.

    Args:
        satellite: one of 'A', 'B' and 'C'.
        collection: SwarmRequest available collections.
        orbit_number:
        download_type: one of None,'measurements','auxiliaries' and 'igrf'.
        None means the download DataFrame only have 'lon','lat','alt','spacecraft' columns and time index.

    Returns:
    """
    download_st = time.perf_counter()

    parts = collection.split("_")
    assert satellite in parts[2], "the spacecraft doesn't match the collection"

    assert download_type in (
        None,
        "measurements",
        "auxiliaries",
        "igrf",
    ), "type must be one of None,'measurements','auxiliaries','igrf'"

    request = SwarmRequest()
    request.set_collection(collection)
    start_time, end_time = request.get_times_for_orbits(
        start_orbit=orbit_number, end_orbit=orbit_number, spacecraft=satellite
    )

    # Store path (stored in b310 server by default).
    if download_type is None:
        sfn = Path(
            f"only_gdcoors_{collection}_{orbit_number}_{start_time.strftime('%Y%m%dT%H%M%S')}_{end_time.strftime('%Y%m%dT%H%M%S')}.pkl"
        )
        request.set_products()
    elif download_type == "measurements":
        sfn = Path(
            f"{collection}_{orbit_number}_{start_time.strftime('%Y%m%dT%H%M%S')}_{end_time.strftime('%Y%m%dT%H%M%S')}.pkl"
        )
        request.set_products(measurements=request.available_measurements(collection))
    elif download_type == "auxiliaries":
        sfn = Path(
            f"aux_{collection}_{orbit_number}_{start_time.strftime('%Y%m%dT%H%M%S')}_{end_time.strftime('%Y%m%dT%H%M%S')}.pkl"
        )
        request.set_products(auxiliaries=SWARM_REQUEST_CONFIG["auxiliaries"])
    else:
        sfn = Path(
            f"IGRF_{collection}_{orbit_number}_{start_time.strftime('%Y%m%dT%H%M%S')}_{end_time.strftime('%Y%m%dT%H%M%S')}.pkl"
        )
        request.set_products(models=["IGRF"])

    if Path(sdir / sfn).exists():
        print(f"文件已存在，跳过下载: {Path(sdir / sfn)}")
        return
    if not sdir.exists():
        sdir.mkdir(parents=True, exist_ok=True)
        print(f"目录已创建: {sdir}")

    # Download data
    try:
        print(f"Start downloading {orbit_number} {collection}:")
        print(f"The customized file name is {sfn}.")

        data = request.get_between(start_time, end_time)
        df = data.as_dataframe()
        df.to_pickle(sdir / sfn)
        download_et = time.perf_counter()

        print(
            f"Successfully download. {sdir / sfn}, cost {download_et - download_st} seconds."
        )
        print(f"The data saved in {sdir/sfn}.")
        print(
            f"Start time: {download_st}.\nEnd time:{download_et}.\ncost time: {download_et-download_st} seconds."
        )
    except Exception as e:
        print(f"An error occurred during download: {e}")
