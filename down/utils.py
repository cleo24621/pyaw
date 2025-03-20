# -*- coding: utf-8 -*-
"""
@Author      : 13927
@Date        : 3/18/2025 16:55
@Project     : pyaw
@Description : 基本变量和函数。

@Copyright   : Copyright (c) 2025, 13927
@License     : 使用的许可证（如 MIT, GPL）
@Last Modified By : 13927
"""


import os
import time
import traceback
from pathlib import Path

from viresclient import SwarmRequest
import logging


auxiliaries=['AscendingNodeLongitude','QDLat', 'QDLon','QDBasis', 'MLT','SunDeclination']
# can use request.get_orbit_number()获取指定时间的轨道号


def download_orbit_collection(spacecraft, collection, orbit_number, type="measurements", mission='Swarm'):
    download_st = time.time()
    assert type in (None,"measurements","auxiliaries","igrf"), "type must be one of None,'measurements','auxiliaries','igrf'"
    request = SwarmRequest()
    request.set_collection(collection)
    st,et = request.get_times_for_orbits(orbit_number, orbit_number, spacecraft=spacecraft, mission=mission)
    # store path
    if type == None:
        sdir = Path(f"V:/aw/swarm/vires/gdcoors/{collection}")
        sfn = Path(f"only_gdcoors_{collection}_{orbit_number}_{st.strftime('%Y%m%dT%H%M%S')}_{et.strftime('%Y%m%dT%H%M%S')}.pkl")
        request.set_products()
    elif type == "measurements":
        sdir = Path(f"V:/aw/swarm/vires/measurements/{collection}")
        sfn = Path(f"{collection}_{orbit_number}_{st.strftime('%Y%m%dT%H%M%S')}_{et.strftime('%Y%m%dT%H%M%S')}.pkl")
        request.set_products(measurements=request.available_measurements(collection))
    elif type == "auxiliaries":
        sdir = Path(f"V:/aw/swarm/vires/auxiliaries/{collection}")
        sfn = Path(f"aux_{collection}_{orbit_number}_{st.strftime('%Y%m%dT%H%M%S')}_{et.strftime('%Y%m%dT%H%M%S')}.pkl")
        request.set_products(auxiliaries=auxiliaries)
    else:
        sdir = Path(f"V:/aw/swarm/vires/igrf/{collection}")
        sfn = Path(f"IGRF_{collection}_{orbit_number}_{st.strftime('%Y%m%dT%H%M%S')}_{et.strftime('%Y%m%dT%H%M%S')}.pkl")
        request.set_products(models=['IGRF'])
    if not sdir.exists():
        sdir.mkdir(parents=True, exist_ok=True)
        print(f"目录已创建: {sdir}")
    if Path(sdir / sfn).exists():
        print(f"文件已存在，跳过下载: {Path(sdir / sfn)}")
        return
    try:
        data = request.get_between(st, et)
        df = data.as_dataframe()
        df.to_pickle(sdir/sfn)
        download_et = time.time()
        # 记录下载信息
        logging.info(
            f"Downloaded: {sfn}, "
            f"Size: {os.path.getsize(Path(sdir / sfn))} bytes, "
            f"Path: {sdir}, "
            f"Time: {download_et - download_st}"
        )
        print(f"download {sdir / sfn}, cost {download_et-download_st}")
    except Exception as e:
        # 创建错误存储目录（若不存在）
        error_dir = sdir / "error_logs"  # 指定错误目录路径
        error_dir.mkdir(parents=True, exist_ok=True)

        # 构建错误文件名（覆写模式）
        error_filename = error_dir / f"error_{st.strftime('%Y%m%dT%H%M%S')}.log"  # 带时间戳的文件名
        # 或者使用固定文件名（每次覆盖）：error_filename = error_dir / "latest_error.log"

        # 将错误信息写入文件（覆写模式）
        with open(error_filename, 'w') as f:
            f.write(f"Error occurred while downloading {orbit_number} {collection}: {e}\n")
            traceback.print_exc(file=f)  # 写入完整堆栈跟踪

        # # 可选：保留原有日志记录（如果需要同时记录到日志文件）
        # logging.error(f"Error occurred while downloading orbit collection: {e}", exc_info=True)


def main():
    pass


if __name__ == "__main__":
    main()
