import numpy as np
from nptyping import NDArray


def get_nor_sou_split_indices(array: NDArray):
    """
    get the indices that split array into northern and southern.
    todo: 也许需要添加处理纬度等于0时属于北半球或南半球的情况（会有吗？（极端情况：0.00001））
    Args:
        array: latitudes. pattern is [south,north], or [south,north,south(few)]

    """
    neg_indices = np.where(array < 0)[0]
    if not neg_indices.size:
        return (0, len(array)), (len(array), len(array))

    start_south = neg_indices[0]
    pos_indices = np.where(array[start_south:] >= 0)[0]
    end_south = start_south + pos_indices[0] if pos_indices.size else len(array)

    return (0, start_south), (start_south, end_south)