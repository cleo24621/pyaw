# -*- coding: utf-8 -*-
"""
@Author      : 13927
@Date        : 3/18/2025 20:23
@Project     : pyaw
@Description : 按轨道下载 mag_lr 产品。

@Copyright   : Copyright (c) 2025, 13927
@License     : 使用的许可证（如 MIT, GPL）
@Last Modified By : 13927
"""


from utils import download_orbit_collection, auxiliaries

collections = ["SW_OPER_MAGA_LR_1B","SW_OPER_MAGB_LR_1B","SW_OPER_MAGC_LR_1B"]


def main():
    pass


if __name__ == "__main__":
    ###
    spacecraft = 'A'
    orbit_number = 12728
    collection = 'SW_OPER_MAGA_LR_1B'
    type = "igrf"  # None,'measurements','auxiliaries','igrf'
    ###
    parts = collection.split('_')
    assert spacecraft in parts[2], "the spacecraft doesn't match the collection"

    download_orbit_collection(spacecraft, collection, orbit_number,type)

    # download orbits by loop orbit_number
