# -*- coding: utf-8 -*-
"""
@Author      : 13927
@Date        : 3/18/2025 21:58
@Project     : pyaw
@Description : 描述此文件的功能和用途。

@Copyright   : Copyright (c) 2025, 13927
@License     : 使用的许可证（如 MIT, GPL）
@Last Modified By : 13927
"""
import numpy as np


def get_split_indices(array):
    """
    Split array into northern and southern indices
    """
    neg_indices = np.where(array < 0)[0]
    if not neg_indices.size:
        return (0, len(array)), (len(array), len(array))

    start_south = neg_indices[0]
    pos_indices = np.where(array[start_south:] >= 0)[0]
    end_south = start_south + pos_indices[0] if pos_indices.size else len(array)

    return (0, start_south), (start_south, end_south)


def main():
    pass


if __name__ == "__main__":
    main()
