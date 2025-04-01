import aacgmv2
import datetime


def trace_to_110km(lat0, lon0, alt0, date):
    """
    沿着磁力线从给定点追踪到110公里高度。

    参数:
        lat (float): 地理纬度（度，北纬为正）
        lon (float): 地理经度（度，东经为正）
        alt (float): 高度（公里，>110）
        date (datetime.date): 日期（用于磁场模型）

    返回:
        (float, float): 110公里处的地理纬度和经度
    """
    # 将地理坐标转换为 CGM 坐标
    mlat, mlon, _ = aacgmv2.get_aacgm_coord(lat0, lon0, alt0, date)

    # 在110公里高度处，将 CGM 坐标转换回地理坐标
    geo_lat, geo_lon, _ = aacgmv2.convert_latlon(mlat, mlon, 110, date, method_code='A2G')

    return geo_lat, geo_lon


if __name__ == "__main__":
    # 设置起始点：纬度 40°N，经度 -75°E，高度 200 km
    start_lat = -70.0
    start_lon = -75.0
    start_alt = 500.0
    # 设置日期：2023年10月1日
    date = datetime.date(2023, 10, 1)

    # 调用函数追踪到110公里处
    end_lat, end_lon = trace_to_110km(start_lat, start_lon, start_alt, date)

    # 输出结果
    print(f"起始点: 纬度={start_lat}°, 经度={start_lon}°, 高度={start_alt} km")
    print(f"追踪到110 km处: 纬度={end_lat:.2f}°, 经度={end_lon:.2f}°")