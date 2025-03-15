# -*- coding: utf-8 -*-
"""
@File: disturb_mag_field_orbits.py
@Author: cleo.py
@Date: 3/6/2025 11:50 PM
@Project: pyaw
@Description: 

@Copyright: Copyright (c) 2025, cleo
@License: (like MIT, GPL ...)
@Last Modified By: cleo
@Last modified Date: 3/6/2025 11:50 PM
"""




def main():
    import numpy as np
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    # 生成卫星轨道数据（实际测量点）
    theta = np.linspace(0, 2 * np.pi, 30)  # 减少采样点数量
    orbit_lons = 30 * np.cos(theta)  # 经度坐标（示例值）
    orbit_lats = 80 + 10 * np.sin(theta)  # 纬度坐标（示例值）

    # 在轨道点上生成电场分量（示例数据）
    E_along = 0.2 * np.cos(theta)  # 沿轨道方向分量
    E_cross = 0.15 * np.sin(2 * theta)  # 横向分量

    # 正则化矢量长度
    magnitude = np.sqrt(E_along ** 2 + E_cross ** 2)
    max_mag = np.max(magnitude)
    scale_factor = 2.0  # 调整缩放系数
    E_along_norm = (E_along / max_mag) * scale_factor
    E_cross_norm = (E_cross / max_mag) * scale_factor

    # 创建极区投影
    proj = ccrs.Orthographic(central_longitude=0, central_latitude=90)

    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection=proj)
    ax.set_extent([-180, 180, 60, 90], crs=ccrs.PlateCarree())

    # 添加地理特征
    ax.add_feature(cfeature.LAND.with_scale('50m'), edgecolor='black', facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='aliceblue')
    ax.coastlines(resolution='50m', linewidth=0.5)

    # 绘制卫星轨道（蓝色实线）
    ax.plot(orbit_lons, orbit_lats, color='blue', linewidth=2,
            transform=ccrs.PlateCarree(), zorder=2)
    ax.scatter(orbit_lons, orbit_lats, color='blue', s=20, zorder=3)

    # 在轨道点上绘制矢量场（关键修改点）
    q = ax.quiver(orbit_lons,  # 使用轨道经度作为X坐标
                  orbit_lats,  # 使用轨道纬度作为Y坐标
                  E_cross_norm,  # 东向分量
                  E_along_norm,  # 北向分量
                  scale=15,  # 调整整体缩放
                  width=0.003,  # 箭杆宽度
                  headwidth=3,  # 减小箭头宽度
                  headlength=4,  # 减小箭头长度
                  headaxislength=3.5,  # 调整箭头头部比例
                  color='red',
                  transform=ccrs.PlateCarree(),
                  zorder=4)

    # 添加矢量标尺
    ax.quiverkey(q, X=0.85, Y=0.15, U=0.2,
                 label='0.2 GV/m', labelpos='E',
                 coordinates='axes')

    # 配置网格和标注
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                      xlocs=range(-180, 181, 30), ylocs=range(60, 91, 10))
    gl.top_labels = False
    gl.right_labels = False

    plt.title('Geo-coordinate System with Regularized Vectors on Orbit\n(22-Jan-2020 02:56:03 to 14:04:42)',
              fontsize=12, pad=20)
    plt.show()


if __name__ == "__main__":
    main()
