import aacgmv2
import numpy as np
from geopack import geopack
from spacepy import coordinates as coord
from spacepy.time import Ticktock


def trace_dip(
        time: np.datetime64,
        alt0,
        lat0,
        lon0,
        _hemisphere,
        trace_set_rlim: int = 999,
        trace_set_maxloop: int = 100000,
        Re=6371.2,
):
    """
    由于geopack的追踪步长不能自定义，所以对于投影极区而言，geopack的效果不好。改用accgmv2.
    Args:
        alt0, lat0, lon0: position of start point. unit are km, degree, degree.
        lon0:
        lat0:
        alt0:
        Re:
        trace_set_maxloop:
        trace_set_rlim: geopack.trace()追踪的最大距离为999个地球半径
        _hemisphere: "N" or "S"
        time: case start time or end time

    Returns:

    """
    assert _hemisphere in ["N", "S"]
    # unify use geo
    c = coord.Coords(
        [(alt0 + Re) / Re, lat0, lon0], "GEO", "sph", ticks=Ticktock(str(time), "ISO")
    )
    # convert geo2gsm
    c_gsm = c.convert("GSM", "car")
    ut = time.astype("datetime64[s]").astype(np.int64)
    _ = geopack.recalc(ut)
    # direction: in north hemisphere, find dip point will have the direction of 由北向南 -> direction = 1 (corresponding to dir of trace() func)
    if _hemisphere == "N":
        direction = 1
    else:
        direction = -1
    # trace: 追踪到地表或者达到追踪的极限停止。x1,y1,z1为最后一个点的3个坐标；xx,yy,zz为所有追踪点的3个坐标组成的3个数组（类型为ndarray）
    x1, x2, x3, xx, yy, zz = geopack.trace(
        c_gsm.data[0, 0],
        c_gsm.data[0, 1],
        c_gsm.data[0, 2],
        dir=direction,
        rlim=trace_set_rlim,
        maxloop=trace_set_maxloop,
    )
    loop_num = len(xx) - 1
    if not loop_num > 0:
        return loop_num, None, None, None, None
    # GSM -> GEO
    c_back = coord.Coords(
        np.column_stack((xx, yy, zz)),
        "GSM",
        "car",
        ticks=Ticktock([str(time) for _ in range(len(xx))], "ISO"),
    )
    c_GEO = c_back.convert("GEO", "sph")  # 返回alt(km),lat(degree),'lon(degree)'
    trace_max_R = np.max(c_GEO.data[:, 0])  # unit: Re
    alts = (c_GEO.data[:, 0] - 1) * Re  # km
    if alts[-1] > 0:
        return loop_num, trace_max_R, None, None, None
    lats = c_GEO.data[:, 1]
    lons = c_GEO.data[:, 2]
    # get dip point position
    if _hemisphere == "N":
        lats_hemi_mask = lats < 0
    else:
        lats_hemi_mask = lats > 0
    lats_hemi = lats[lats_hemi_mask]
    alts_hemi = alts[lats_hemi_mask]
    lons_hemi = lons[lats_hemi_mask]
    idx = np.argmin(np.abs(alts_hemi - alt0))

    return loop_num, trace_max_R, alts_hemi[idx], lats_hemi[idx], lons_hemi[idx]


def trace_auro_with_aacgmv2(lat0, lon0, alt0, date, auroral_altitude: float = 110):
    """
    沿着磁力线从给定点追踪到auroral_altitude公里高度。

    参数:
        lat (float): 地理纬度（度，北纬为正）
        lon (float): 地理经度（度，东经为正）
        alt (float): 高度（公里，>auroral_altitude）
        date (datetime.date): 日期（用于磁场模型）

    返回:
        (float, float): auroral_altitude公里处的地理纬度和经度

    Examples:
        end_lat, end_lon = trace_to_110km(70, 0, 500, datetime.date(2023, 10, 1))
    """
    assert alt0 > auroral_altitude
    # 将地理坐标转换为 CGM 坐标
    mlat, mlon, _ = aacgmv2.get_aacgm_coord(lat0, lon0, alt0, date)

    # 在110公里高度处，将 CGM 坐标转换回地理坐标
    geo_lat, geo_lon, _ = aacgmv2.convert_latlon(
        mlat, mlon, auroral_altitude, date, method_code="A2G"
    )

    return geo_lat, geo_lon
