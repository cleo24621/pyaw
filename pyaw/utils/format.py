import numpy as np
from datetime import datetime


def timestr2datetime64ns(time_string: str) -> np.datetime64:
    """Converts Swarm-style timestamp string to numpy datetime64 with nanosecond precision.

    Args:
        time_string: Timestamp string in format 'YYYYMMDDTHHMMSS' (e.g. '20160311T064700')

    Returns:
        np.datetime64: Datetime object with nanosecond precision

    Example:
        >>> timestr2datetime64ns('20160311T064700')
        numpy.datetime64('2016-03-11T06:47:00.000000000')
    """
    dt_obj = datetime.strptime(time_string, "%Y%m%dT%H%M%S")
    return np.datetime64(dt_obj, "ns")
