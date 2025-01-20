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


# request = SwarmRequest()
# st = time.time()
# orbit_number_st_et = 12885
# orbit_st, orbit_et = request.get_times_for_orbits(orbit_number_st_et, orbit_number_st_et, mission='Swarm', spacecraft='A')
# tct_vars = [  # Satellite velocity in NEC frame
#         "VsatC", "VsatE", "VsatN",  # Geomagnetic field components derived from 1Hz product
#         #  (in satellite-track coordinates)
#         "Bx", "By", "Bz",  # Electric field components derived from -VxB with along-track ion drift
#         #  (in satellite-track coordinates)
#         # Eh: derived from horizontal sensor
#         # Ev: derived from vertical sensor
#         "Ehx", "Ehy", "Ehz", "Evx", "Evy", "Evz",
#         # Ion drift corotation signal, removed from ion drift & electric field
#         #  (in satellite-track coordinates)
#         "Vicrx", "Vicry", "Vicrz",  # Ion drifts along-track from vertical (..v) and horizontal (..h) TII sensor
#         "Vixv", "Vixh",  # Ion drifts cross-track (y from horizontal sensor, z from vertical sensor)
#         #  (in satellite-track coordinates)
#         "Viy", "Viz",  # Random error estimates for the above
#         #  (Negative value indicates no estimate available)
#         "Vixv_error", "Vixh_error", "Viy_error", "Viz_error",  # Quasi-dipole magnetic latitude and local time
#         #  redundant with VirES auxiliaries, QDLat & MLT
#         "Latitude_QD", "MLT_QD",  # Refer to release notes link above for details:
#         "Calibration_flags", "Quality_flags", ]
# request.set_collection(f"SW_EXPT_EFIA_TCT16")
# request.set_products(measurements=tct_vars)
# data = request.get_between(orbit_st, orbit_et)
# df = data.as_dataframe()
# df.to_pickle("./data/result/12885_SW_EXPT_EFIA_TCT16.pkl")
# et = time.time()
# print("Time cost: ", et-st)

# 通过request的get_between()下载的数据都默认有lat,lon,radius,spacecraft数据。
# 通过set_colletion()设置多个colletions，默认按照第一个colletion的时间获取对应的数据。
# 就算是同样频率的数据产品，其时间数据也不一定相等。例如"SW_EXPT_EFIA_TCT02"和"SW_OPER_EFIA_LP_1B"。
# 确定需要的卫星：ABC
# 确定需要的数据产品：
# SW_EXPT_EFIx_TCT16
# SW_OPER_MAGx_HR_1B
# 等离子体密度（电子）和等离子体电子温度：SW_OPER_EFIx_LP_1B
# 确定年份
# 2016

def download_orbit_collection(request,spacecraft,orbit_number,collection):
    """
    下载单轨数据产品，并保存到指定路径
    :return:
    """
    st,et = request.get_times_for_orbits(orbit_number, orbit_number, spacecraft=spacecraft, mission='Swarm')  # todo: 将轨道对应的起止时间存储起来，不用重复获取时间
    sdir = Path(f"V:/aw/swarm/vires/{collection}")
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

# 要循环的列表
request = SwarmRequest()
orbit_number_st_et = {'A':(request.get_orbit_number('A', '20160101T000000', mission='Swarm'),
                           request.get_orbit_number('A','20170101T000000',mission='Swarm')),
                'B': (request.get_orbit_number('B', '20160101T000000', mission='Swarm'),
                      request.get_orbit_number('B', '20170101T000000', mission='Swarm')),
                'C': (request.get_orbit_number('C', '20160101T000000', mission='Swarm'),
                      request.get_orbit_number('C', '20170101T000000', mission='Swarm')),
                      }
collections_dic = {'TCT16':["SW_EXPT_EFIA_TCT16", "SW_EXPT_EFIB_TCT16", "SW_EXPT_EFIC_TCT16"],
               'MAG_HR':["SW_OPER_MAGA_HR_1B","SW_OPER_MAGB_HR_1B","SW_OPER_MAGC_HR_1B"],
               'LP_1B':["SW_OPER_EFIA_LP_1B","SW_OPER_EFIB_LP_1B","SW_OPER_EFIC_LP_1B"]}

for collection_key,collection_value_ls in collections_dic.items():
    print(collection_key)
    print(collection_value_ls)
    for collection,(spacecraft, orbit_number_st_et_value) in zip(collection_value_ls, orbit_number_st_et.items()):
        print(collection)
        print(spacecraft)
        print(orbit_number_st_et_value)
        request.set_collection(collection)
        measurements = request.available_measurements(collection)
        request.set_products(measurements=measurements)
        for orbit_number in range(orbit_number_st_et_value[0],orbit_number_st_et_value[1]+1):
            print(orbit_number)
            try:
                download_st = time.time()
                download_orbit_collection(request, spacecraft, orbit_number, collection)
                time.sleep(1)
                download_et = time.time()
                print(f"download cost: {download_et-download_st} s")
            except Exception as e:
                # 记录错误信息
                logging.error(f"Error occurred while downloading orbit collection: {e}", exc_info=True)





# 一轨数据下载
# 极光带 普遍现象？ Wu 2020研究结论 是普遍现象