import time
from viresclient import SwarmRequest
import logging

from vires_down_utils import download_orbit_collection


# 要循环的列表
request = SwarmRequest()
# orbit_number_st_et = {'A':(request.get_orbit_number('A', '20160101T000000', mission='Swarm'),
#                            request.get_orbit_number('A','20170101T000000',mission='Swarm')),
#                 'B': (request.get_orbit_number('B', '20160101T000000', mission='Swarm'),
#                       request.get_orbit_number('B', '20170101T000000', mission='Swarm')),
#                 'C': (request.get_orbit_number('C', '20160101T000000', mission='Swarm'),
#                       request.get_orbit_number('C', '20170101T000000', mission='Swarm')),
#                       }
# collections_dic = {'TCT16':["SW_EXPT_EFIA_TCT16", "SW_EXPT_EFIB_TCT16", "SW_EXPT_EFIC_TCT16"],
#                'MAG_HR':["SW_OPER_MAGA_HR_1B","SW_OPER_MAGB_HR_1B","SW_OPER_MAGC_HR_1B"],
#                'LP_1B':["SW_OPER_EFIA_LP_1B","SW_OPER_EFIB_LP_1B","SW_OPER_EFIC_LP_1B"]}

orbit_number_st_et = {'A':(request.get_orbit_number('A', '20160301T000000', mission='Swarm'),
                           request.get_orbit_number('A','20160501T000000',mission='Swarm')),
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