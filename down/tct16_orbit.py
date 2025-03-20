# -*- coding: utf-8 -*-
"""
@Author      : 13927
@Date        : 3/18/2025 17:18
@Project     : pyaw
@Description : 按轨道下载 tct16 产品。

@Copyright   : Copyright (c) 2025, 13927
@License     : 使用的许可证（如 MIT, GPL）
@Last Modified By : 13927
"""


from utils import download_orbit_collection, auxiliaries

collections = ["SW_EXPT_EFIA_TCT16","SW_EXPT_EFIB_TCT16","SW_EXPT_EFIC_TCT16"]


def main():
    pass


if __name__ == "__main__":
    ###
    spacecraft = 'A'
    orbit_number = 12019
    collection = 'SW_EXPT_EFIA_TCT16'
    type = None
    ###
    parts = collection.split('_')
    assert spacecraft in parts[2], "the spacecraft doesn't match the collection"

    download_orbit_collection(spacecraft, collection, orbit_number,type)

    # download orbits by loop orbit_number