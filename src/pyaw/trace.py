import datetime as dt

import aacgmv2
import numpy as np
from astropy.constants import R_earth
from geopack import geopack
from spacepy import coordinates as coord
from spacepy.time import Ticktock

from configs import RLIM, MAXLOOP


def geo2gsm_by_spacepy(glat: float, glon: float, height: float, dtime: dt.datetime):
    c = coord.Coords(
        [(height + R_earth) / R_earth, glat, glon],
        "GEO",
        "sph",
        ticks=Ticktock(str(dtime), "ISO"),
    )

    return c.convert("GSM", "car")


def trace_by_geopack(
    dtime: dt.datetime,
    dir,
    c_gsm: coord.Coords,
    rlim: float = RLIM,
    maxloop: int = MAXLOOP,
):
    """Trace a field line using geopack."""
    # _ = geopack.recalc(dtime.timestamp)
    x1, x2, x3, xx, yy, zz = geopack.trace(
        c_gsm.data[0, 0],
        c_gsm.data[0, 1],
        c_gsm.data[0, 2],
        dir=dir,
        rlim=rlim,
        maxloop=maxloop,
    )
    c_back = coord.Coords(
        np.column_stack((xx, yy, zz)),
        "GSM",
        "car",
        ticks=Ticktock([str(dtime) for _ in range(len(xx))], "ISO"),
    )
    c_geo = c_back.convert("GEO", "sph")  # height (km), glat (deg N), glon (deg E)

    return c_geo


def trace2auroral_height_by_aacgmv2(
    glat: float,
    glon: float,
    height: float,
    dtime: dt.datetime,
    auroral_height: float = 110,
) -> tuple:
    """Get the latitude, longitude and height of the point (glat, glon, height) at the auroral height using AACGMv2.

    Args:
        glat: Geodetic latitude in degrees N
        glon: Geodetic longitude in degrees E
        height: Altitude above the surface of the earth in km
        dtime: Date and time to calculate magnetic location
        auroral_height: Altitude of the aurora in km
    """
    mlat, mlon, mlt = aacgmv2.get_aacgm_coord(glat, glon, height, dtime)
    out_glat, out_glon, out_height = aacgmv2.convert_latlon(
        mlat, mlon, auroral_height, dtime, method_code="A2G"
    )

    return out_glat, out_glon, out_height
