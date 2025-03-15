# -*- coding: utf-8 -*-
"""
@File: vires_down_loop_igrf.py
@Author: cleo.py
@Date: 2025/2/26 10:20
@Project: pyaw
@Description: 

@Copyright: Copyright (c) 2025, cleo
@License: (like MIT, GPL ...)
@Last Modified By: cleo
@Last modified Date: 2025/2/26 10:20
"""
import os
import time
from pathlib import Path

from viresclient import SwarmRequest
import logging


def download_orbit_collection(request,spacecraft,orbit_number,collection,sdir="V:/aw/swarm/vires"):
    """
    下载单轨数据产品，并保存到指定路径
    :return:
    """
    st,et = request.get_times_for_orbits(orbit_number, orbit_number, spacecraft=spacecraft, mission='Swarm')  # todo: 将轨道对应的起止时间存储起来，不用重复获取时间
    # sdir = Path(sdir) / Path(f"{collection}")
    # sdir = Path(sdir) / Path(f"IGRF/{collection}")
    sdir = Path(f"V:/aw/swarm/vires/{collection}")
    sfn = Path(f"aux_{collection}_{orbit_number}_{st.strftime('%Y%m%dT%H%M%S')}_{et.strftime('%Y%m%dT%H%M%S')}.pkl")
    # sfn = Path(f"IGRF_{collection}_{orbit_number}_{st.strftime('%Y%m%dT%H%M%S')}_{et.strftime('%Y%m%dT%H%M%S')}.pkl")
    # sfn = Path(f"{collection}_{orbit_number}_{st.strftime('%Y%m%dT%H%M%S')}_{et.strftime('%Y%m%dT%H%M%S')}.pkl")
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


# 要循环的列表
request = SwarmRequest()

orbit_number_st_et = {'A':(request.get_orbit_number('A', '20160601T000000', mission='Swarm'),
                           request.get_orbit_number('A','20160701T000000',mission='Swarm')),
                      }
collections_dic = {'MAG_HR':["SW_OPER_MAGA_HR_1B"]}
# collections_dic = {'TCT16':["SW_EXPT_EFIA_TCT16"]}


for collection_key,collection_value_ls in collections_dic.items():
    print(collection_key)
    print(collection_value_ls)
    for collection,(spacecraft, orbit_number_st_et_value) in zip(collection_value_ls, orbit_number_st_et.items()):
        print(collection)
        print(spacecraft)
        print(orbit_number_st_et_value)
        request.set_collection(collection)
        measurements = request.available_measurements(collection)
        # request.set_products(auxiliaries=['AscendingNodeLongitude','QDLat', 'QDLon','QDBasis', 'MLT','SunDeclination'])
        # request.set_products(models=['IGRF'])
        request.set_products(measurements=measurements)
        for orbit_number in range(orbit_number_st_et_value[0],orbit_number_st_et_value[1]+1):
            print(orbit_number)
            try:
                download_st = time.time()
                download_orbit_collection(request, spacecraft, orbit_number, collection)
                # time.sleep(1)
                download_et = time.time()
                print(f"download cost: {download_et-download_st} s")
            except Exception as e:
                # 记录错误信息
                logging.error(f"Error occurred while downloading orbit collection: {e}", exc_info=True)

def main():
    pass


if __name__ == "__main__":
    main()
