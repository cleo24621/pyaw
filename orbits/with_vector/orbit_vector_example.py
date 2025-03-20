import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 生成覆盖北半球的卫星轨道数据
theta = np.linspace(0, 2 * np.pi, 30)
orbit_lons = 50 * np.cos(theta)          # 经度跨度 ±50°
orbit_lats = 45 + 45 * np.sin(theta)

# 计算轨道切向角度（使用中央差分法）
d_lon = np.gradient(orbit_lons, theta)
d_lat = np.gradient(orbit_lats, theta)
alpha = np.arctan2(d_lat, d_lon)  # 轨道切向与东向夹角

# 生成示例电场数据（与纬度相关）
E_along = 0.3 * np.cos(theta) * np.sin(0.5*theta)  # 沿轨道分量
E_cross = 0.2 * np.sin(2*theta) * (orbit_lats/90)  # 横向分量（纬度调制）

# 分解到地理坐标系
E_east = E_along * np.cos(alpha) - E_cross * np.sin(alpha)
E_north = E_along * np.sin(alpha) + E_cross * np.cos(alpha)

# 正则化矢量（保留相对比例）
max_mag = np.max(np.sqrt(E_east**2 + E_north**2))
scale_factor = 1.8
E_east_norm = (E_east / max_mag) * scale_factor
E_north_norm = (E_north / max_mag) * scale_factor

# 创建极区投影并设置范围
proj = ccrs.Orthographic(central_longitude=0, central_latitude=90)
geodetic = ccrs.PlateCarree()

# 转换矢量到投影坐标系
u_proj, v_proj = proj.transform_vectors(
    geodetic, orbit_lons, orbit_lats, E_east_norm, E_north_norm
)

# 绘制完整北半球
plt.figure(figsize=(12, 12))
ax = plt.axes(projection=proj)
ax.set_extent([-180, 180, 10, 90], crs=geodetic)  # 关键修改：纬度下限设为0°

# 添加地理特征
ax.add_feature(cfeature.LAND.with_scale('50m'), alpha=0.6)
ax.add_feature(cfeature.OCEAN.with_scale('50m'), alpha=0.4)
ax.coastlines(resolution='50m', linewidth=0.5)

# 绘制轨道和矢量
ax.plot(orbit_lons, orbit_lats, color='navy', linewidth=1.5,
        transform=geodetic, zorder=2)
ax.scatter(orbit_lons, orbit_lats, color='navy', s=15, zorder=3)

q = ax.quiver(
    orbit_lons, orbit_lats, u_proj, v_proj,
    scale=35,         # 增大scale值以适应更大范围
    width=0.0025,     # 更细的箭头杆
    headwidth=4,
    headlength=5,
    color='crimson',
    transform=geodetic,
    zorder=4
)

# 配置网格线
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                 xlocs=range(-180, 181, 45),
                 ylocs=range(0, 91, 15))  # 纬度标签从0°开始
gl.top_labels = False
gl.right_labels = False
gl.rotate_labels = False

# 添加矢量标尺
ax.quiverkey(q, X=0.82, Y=0.12, U=0.3,
             label='Normalized Vector', labelpos='E',
             coordinates='axes')

plt.title('North Hemisphere Electric Field Vectors along Satellite Orbit\n'
          '(Latitude Range: 0°–90°, Time: 2020-01-22 02:56–14:04 UTC)',
          fontsize=12, pad=18)
plt.show()