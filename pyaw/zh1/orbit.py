import numpy as np
from nptyping import NDArray


def get_orbitnumber_indicator_st_et(file_name):
    """
    因为不同2级产品的命名格式是固定的，所以当前方法适用于所有2a级产品的文件名（参考文档）。
    1: ascending (south to north)
    0: descending (north to south)
    Args:
        file_name: now only support the efd 2a data filename
    """
    parts = file_name.split("_")
    part = parts[6]
    assert part[-1] in ["0", "1"]
    start_time = parts[7] + "_" + parts[8]
    end_time = parts[9] + "_" + parts[10]
    return parts[6][:-1], parts[6][-1], start_time, end_time


def get_nor_sou_split_indices(array: NDArray, indicator: str):
    """

    Args:
        array: latitudes
        indicator: 对于zh1而言，indicator="0" or indicator="1"

    Returns:

    """
    assert indicator in ["1", "0"]
    if all(array > 0):
        return (0, len(array)), (len(array), len(array))
    elif all(array < 0):
        return (len(array), len(array)), (0, len(array))
    elif indicator == "1":
        start_north = np.where(array > 0)[0][0]
        return (start_north, None), (0, start_north)  # north, south slice
    else:
        start_south = np.where(array < 0)[0][0]
        return (0, start_south), (start_south, None)  # north, south slice
