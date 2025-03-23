# -*- coding: utf-8 -*-
"""
@Author      : 13927
@Date        : 3/21/2025 0:01
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
    # orbit
    spacecraft = 'A'
    orbit_number = 12019
    collection = 'SW_EXPT_EFIA_TCT16'
    download_type = None
    SwarmDownload.download_orbit_collection(spacecraft,collection,orbit_number,download_type)

    # # orbits
    # spacecraft = 'A'
    # start_orbit_number, end_orbit_number = 12019,12020
    # collection = 'SW_EXPT_EFIA_TCT16'
    # download_type = None
    # for orbit_number in range(start_orbit_number, end_orbit_number + 1):
    #     SwarmDownload.download_orbit_collection(spacecraft, collection, orbit_number, download_type)