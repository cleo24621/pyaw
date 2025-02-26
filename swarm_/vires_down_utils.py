"""
通过vires下载其支持的数据，主要按照轨道下载，并实现下载过程记录。
"""

import os
import time
from pathlib import Path

from viresclient import SwarmRequest
import logging


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    filename='../download.log',
    filemode='a'
)
logging.basicConfig(
    level=logging.ERROR,  # 设置日志级别为 ERROR
    format='%(asctime)s - %(levelname)s - %(message)s',  # 设置日志格式
    filename='error.log',  # 将日志写入文件
    filemode='a'  # 追加模式
)


def download_orbit_collection(request,spacecraft,orbit_number,collection,sdir="V:/aw/swarm/vires"):
    """
    下载单轨数据产品，并保存到指定路径
    :return:
    """
    st,et = request.get_times_for_orbits(orbit_number, orbit_number, spacecraft=spacecraft, mission='Swarm')  # todo: 将轨道对应的起止时间存储起来，不用重复获取时间
    sdir = Path(sdir) / Path(f"{collection}")
    # sdir = Path(f"V:/aw/swarm/vires/{collection}")
    sfn = Path(f"{collection}_{orbit_number}_{st.strftime('%Y%m%dT%H%M%S')}_{et.strftime('%Y%m%dT%H%M%S')}.pkl")
    if not sdir.exists():
        sdir.mkdir(parents=True, exist_ok=True)
        print(f"目录已创建: {sdir}")
    if Path(sdir/sfn).exists():
        print(f"文件已存在，跳过下载: {Path(sdir/sfn)}")
        return
    download_st = time.time()
    data = request.get_between(st,et)
    df = data.as_dataframe()
    df.to_pickle(sdir/sfn)
    download_et = time.time()
    # 记录下载信息
    logging.info(
        f"Downloaded: {sfn}, "
        f"Size: {os.path.getsize(Path(sdir/sfn))} bytes, "
        f"Path: {sdir}, "
        f"Time: {download_et-download_st}"
    )
    print(f"download {sdir/sfn}")


def main():
    """
    下载单轨数据
    """
    request = SwarmRequest()
    s_c = 'A'
    input_time = '20250101T000000'
    mission = 'Swarm'
    o_n = request.get_orbit_number(s_c,input_time,mission)
    collection = "SW_OPER_MAGA_HR_1B"
    request.set_collection(collection)
    measurements = request.available_measurements(collection)
    request.set_products(measurements=measurements)
    download_orbit_collection(request,s_c,o_n,collection)

if __name__ == "__main__":
    main()