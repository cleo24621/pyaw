# -*- coding: utf-8 -*-
"""
@Author      : 13927
@Date        : 3/21/2025 20:45
@Project     : pyaw
@Description : 和格式化有关。

@Copyright   : Copyright (c) 2025, 13927
@License     : 使用的许可证（如 MIT, GPL）
@Last Modified By : 13927
"""
import numpy as np


def timestr2datetime64ns(time_string: str) -> np.datetime64:
    """
    convert "20160311T064700" type string to np.datetime64[ns] type
    :param time_string: "20160311T064700" type
    :return: np.datetime64[ns]
    """
    # Insert delimiters to make it ISO 8601 compliant
    formatted_string = (
        time_string[:4]
        + "-"
        + time_string[4:6]
        + "-"
        + time_string[6:8]
        + "T"
        + time_string[9:11]
        + ":"
        + time_string[11:13]
        + ":"
        + time_string[13:]
    )
    # Convert to numpy.datetime64 with nanosecond precision
    return np.datetime64(formatted_string, "ns")


def main():
    pass


if __name__ == "__main__":
    main()
