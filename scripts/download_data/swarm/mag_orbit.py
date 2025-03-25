# -*- coding: utf-8 -*-
"""
@Author      : 13927
@Date        : 3/21/2025 0:18
@Project     : pyaw
@Description : 描述此文件的功能和用途。

@Copyright   : Copyright (c) 2025, 13927
@License     : 使用的许可证（如 MIT, GPL）
@Last Modified By : 13927
"""
from pyaw.core.download_data import SwarmDownload


def main():
    pass


if __name__ == "__main__":
    # mag_hr orbit
    spacecraft = 'A'
    orbit_number = 12019
    collection = 'SW_OPER_MAGA_HR_1B'
    download_type = None
    SwarmDownload.download_orbit_collection(spacecraft,collection,orbit_number,download_type)

    # # mag_hr orbits
    # spacecraft = 'A'
    # start_orbit_number, end_orbit_number = 12019,12020
    # collection = 'SW_OPER_MAGA_HR_1B'
    # download_type = None
    # for orbit_number in range(start_orbit_number, end_orbit_number + 1):
    #     SwarmDownload.download_orbit_collection(spacecraft, collection, orbit_number, download_type)

    # # mag_lr orbit
    # spacecraft = 'A'
    # orbit_number = 12019
    # collection = 'SW_OPER_MAGA_LR_1B'
    # download_type = None
    # SwarmDownload.download_orbit_collection(spacecraft, collection, orbit_number, download_type)

    # # mag_lr orbits
    # spacecraft = 'A'
    # start_orbit_number, end_orbit_number = 12019, 12020
    # collection = 'SW_OPER_MAGA_LR_1B'
    # download_type = None
    # for orbit_number in range(start_orbit_number, end_orbit_number + 1):
    #     SwarmDownload.download_orbit_collection(spacecraft, collection, orbit_number, download_type)