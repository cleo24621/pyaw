import os
import time
from pathlib import Path

from viresclient import SwarmRequest
import logging

from configs import ViresSwarmRequest


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
        request.set_products(
            measurements=request.available_measurements(collection)
        )
    elif download_type == "auxiliaries":
        store_dir_path = Path(f"V:/aw/swarm/vires/auxiliaries/{collection}")
        store_file_name = Path(
            f"aux_{collection}_{orbit_number}_{start_time.strftime('%Y%m%dT%H%M%S')}_{end_time.strftime('%Y%m%dT%H%M%S')}.pkl"
        )
        request.set_products(
            auxiliaries=ViresSwarmRequest.auxiliaries
        )
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