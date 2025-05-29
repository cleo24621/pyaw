import logging
import os
import time
from pathlib import Path

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
    """process tct16 l1b production
    for tct16, use 'nan_outliers_by_std_dev()' to process outlier
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.dataframe = pd.read_pickle(file_path)
        self.VsatN = self.dataframe["VsatN"].values
        self.VsatE = self.dataframe["VsatE"].values

    def sc2nec(self, vectorx, vectory):
        f"""vector s/c x,y component change to nec n,e component
        
        Args:
            vectorx: for electric filed, need input -Ex not Ex
            vectory: same as {vectorx}

        Returns:
            a tuple, the first is north component, the second is east component.
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
        vector: NDArray,
        quaternion: NDArray,
    ) -> NDArray:
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

    def _calculate_rotated_vectors(self) -> NDArray:
        """
        Get the variable data in SC Frame. (Note that the SC Frame is different between Swarm and DMSP)
        Args:

        Returns:
            the variable data in SC Frame
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
        spacecraft: str, collection: str, orbit_number: int, download_type: str | None
) -> None:
    """
    Download 1 orbit of collection data as a DataFrame and save as '.pkl' file using 'vires' tool.

    Args:
        spacecraft: Swarm: one of (‘A’,’B’,’C’) or (“Alpha”, “Bravo”, “Charlie”)
        collection:
        orbit_number:
        download_type: one of (None,'measurements','auxiliaries','igrf'). The type of None means that the download
        DataFrame only have 'lon','lat','alt','spacecraft' columns and time index.

    Returns:
        object:
    """
    download_st = time.time()
    parts = collection.split("_")
    assert spacecraft in parts[2], "the spacecraft doesn't match the collection"
    assert download_type in (
        None,
        "measurements",
        "auxiliaries",
        "igrf",
    ), "type must be one of None,'measurements','auxiliaries','igrf'"
    request = SwarmRequest()
    request.set_collection(collection)
    start_time, end_time = request.get_times_for_orbits(
        start_orbit=orbit_number, end_orbit=orbit_number, spacecraft=spacecraft
    )
    # store path (stored in b310 server)
    if download_type is None:
        store_dir_path = Path(f"V:/aw/swarm/vires/gdcoors/{collection}")
        store_file_name = Path(
            f"only_gdcoors_{collection}_{orbit_number}_{start_time.strftime('%Y%m%dT%H%M%S')}_{end_time.strftime('%Y%m%dT%H%M%S')}.pkl"
        )
        request.set_products()
    elif download_type == "measurements":
        store_dir_path = Path(f"V:/aw/swarm/vires/measurements/{collection}")
        store_file_name = Path(
            f"{collection}_{orbit_number}_{start_time.strftime('%Y%m%dT%H%M%S')}_{end_time.strftime('%Y%m%dT%H%M%S')}.pkl"
        )
        request.set_products(measurements=request.available_measurements(collection))
    elif download_type == "auxiliaries":
        store_dir_path = Path(f"V:/aw/swarm/vires/auxiliaries/{collection}")
        store_file_name = Path(
            f"aux_{collection}_{orbit_number}_{start_time.strftime('%Y%m%dT%H%M%S')}_{end_time.strftime('%Y%m%dT%H%M%S')}.pkl"
        )
        request.set_products(auxiliaries=SWARM_REQUEST_CONFIG["auxiliaries"])
    else:
        store_dir_path = Path(f"V:/aw/swarm/vires/igrf/{collection}")
        store_file_name = Path(
            f"IGRF_{collection}_{orbit_number}_{start_time.strftime('%Y%m%dT%H%M%S')}_{end_time.strftime('%Y%m%dT%H%M%S')}.pkl"
        )
        request.set_products(models=["IGRF"])
    if Path(store_dir_path / store_file_name).exists():
        print(f"文件已存在，跳过下载: {Path(store_dir_path / store_file_name)}")
        return
    if not store_dir_path.exists():
        store_dir_path.mkdir(parents=True, exist_ok=True)
        print(f"目录已创建: {store_dir_path}")
    try:
        data = request.get_between(start_time, end_time)
        df = data.as_dataframe()
        df.to_pickle(store_dir_path / store_file_name)
        download_et = time.time()
        # 记录下载信息
        # config log
        log_file_path = store_dir_path / Path("logfile.log")  # 自定义日志文件路径
        logging.basicConfig(
            filename=log_file_path,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        # write log
        logging.info(
            f"Downloaded: {store_file_name}, "
            f"Size: {os.path.getsize(Path(store_dir_path / store_file_name))} bytes, "
            f"Path: {store_dir_path}, "
            f"Time: {download_et - download_st}"
        )
        # print information on the console
        print(
            f"download {store_dir_path / store_file_name}, cost {download_et - download_st}"
        )
    except Exception as e:
        # 创建错误存储目录（若不存在）
        error_dir = store_dir_path / "error_logs"  # 指定错误目录路径
        error_dir.mkdir(parents=True, exist_ok=True)
        # 构建错误文件名（覆写模式）
        error_filename = (
                error_dir / f"error_{start_time.strftime('%Y%m%dT%H%M%S')}.log"
        )  # 带时间戳的文件名
        # 或者使用固定文件名（每次覆盖）：error_filename = error_dir / "latest_error.log"
        # 将错误信息写入文件（覆写模式）
        with open(error_filename, "w") as f:
            f.write(
                f"Error occurred while downloading {orbit_number} {collection}: {str(e)}\n"
            )
        # # 可选：保留原有日志记录（如果需要同时记录到日志文件）
        # logging.error(f"Error occurred while downloading orbit collection: {e}", exc_info=True)
