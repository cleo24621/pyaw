from geopack import geopack
import datetime
import numpy as np

# --- 1. 准备环境和参数 ---
# 设置时间
time = datetime.datetime(2001, 3, 21, 12, 0, 0) # Example time
ut = time.hour * 3600 + time.minute * 60 + time.second + time.microsecond * 1e-6
# 计算偶极子倾角 psi (需要 geopack.recalc)
psi = geopack.recalc(ut) # Note: recalc might need year/day too depending on version

# 设置 T96 模型参数 (示例)
parmod = np.zeros(10)
parmod[0] = -5.0   # Pdyn (nPa)
parmod[1] = 2.0    # Dst (nT)
parmod[2] = 0.0    # By IMF (nT)
parmod[3] = -3.0   # Bz IMF (nT)
# parmod[4] to parmod[9] are W parameters for T01/T04, often 0 for T96

# --- 2. 定义起始点 (从地理坐标 -> GSM) ---
glat_start = 65.0  # Geographic Latitude (degrees)
glon_start = 0.0   # Geographic Longitude (degrees)
alt_start_km = 110.0 # Altitude (km)
Re_km = 6371.2    # Earth radius in km

# 地理球坐标 -> GEO笛卡尔 (注意：geopack.sphcar 可能需要 theta=90-lat)
# 检查 geopack.sphcar 文档：1 for sph->car, -1 for car->sph
# It might expect colatitude (90-lat) or latitude directly. Assuming latitude here.
# Radius in RE
r_start_RE = 1.0 + alt_start_km / Re_km
xgeo_start, ygeo_start, zgeo_start = geopack.sphcar(glat_start, glon_start, r_start_RE, 1)

# GEO -> GSM (需要 psi)
# 检查 geopack.geogsm 文档：1 for GEO->GSM, -1 for GSM->GEO
xgsm_start, ygsm_start, zgsm_start = geopack.geogsm(xgeo_start, ygeo_start, zgeo_start, 1, psi)
print(f"起始点 (GSM): X={xgsm_start:.3f}, Y={ygsm_start:.3f}, Z={zgsm_start:.3f} RE")

# --- 3. 调用 geopack.trace ---
rlim_stop = 1.0  # Stop tracing at Earth's surface (R=1 RE)
# 确定追踪方向：如果 zgsm_start > 0 (北半球), B 大致指向南, dir=1
# 如果 zgsm_start < 0 (南半球), B 大致指向北, dir=-1 (或者用 dir=1 让它反向追踪)
# 简单起见，假设北半球起点，追踪向南
trace_dir = 1 if zgsm_start > 0 else -1

# 执行追踪
try:
    x_trace, y_trace, z_trace = geopack.trace(
        xgsm_start, ygsm_start, zgsm_start,
        dir=trace_dir,
        rlim=rlim_stop,
        r0=r_start_RE, # Provide starting radius
        parmod=parmod,
        exname='t96',
        inname='igrf',
        maxloop=20000 # Increase steps if needed
    )
    print(f"追踪完成，共 {len(x_trace)} 个点。")

    # --- 4. 提取共轭点坐标 ---
    if len(x_trace) > 1:
        xgsm_conj = x_trace[-1]
        ygsm_conj = y_trace[-1]
        zgsm_conj = z_trace[-1]
        R_conj = np.sqrt(xgsm_conj**2 + ygsm_conj**2 + zgsm_conj**2)

        # 验证终点
        if abs(R_conj - rlim_stop) < 0.05 and zgsm_conj * zgsm_start < 0: # 检查半径和半球
            print(f"找到潜在共轭点 (GSM): X={xgsm_conj:.3f}, Y={ygsm_conj:.3f}, Z={zgsm_conj:.3f} RE, R={R_conj:.3f} RE")

            # --- 5. 坐标转换 (GSM -> Geographic) ---
            # GSM -> GEO
            xgeo_conj, ygeo_conj, zgeo_conj = geopack.geogsm(xgsm_conj, ygsm_conj, zgsm_conj, -1, psi)

            # GEO -> Geographic Spherical (lat, lon, alt)
            # 检查 geopack.sphcar 文档 for return order and units (deg or rad)
            R_RE_conj, glat_conj, glon_conj = geopack.sphcar(xgeo_conj, ygeo_conj, zgeo_conj, -1)
            alt_conj_km = (R_RE_conj - 1.0) * Re_km

            print(f"共轭点 (地理): Lat={glat_conj:.2f} deg, Lon={glon_conj:.2f} deg, Alt={alt_conj_km:.1f} km")

        else:
            print(f"追踪终点 (R={R_conj:.3f} RE, Z={zgsm_conj:.3f} RE) 不满足共轭点条件 (R != {rlim_stop} 或在同半球)。可能是开放磁力线或追踪问题。")

    else:
        print("追踪失败，未生成有效路径点。")

except Exception as e:
    print(f"追踪过程中发生错误: {e}")