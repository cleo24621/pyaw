import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

fp = r'D:\cleo\master\pyaw\data\Swarm\aux_SW_OPER_MAGA_HR_1B_12885_20160311T061733_20160311T075106.pkl'
df_b_aux = pd.read_pickle(fp)
longitudes = df_b_aux['Longitude'].values
latitudes = df_b_aux['Latitude'].values

# 数据过滤
# northern_hemisphere_mask = latitudes > 0
# longitudes = longitudes[northern_hemisphere_mask]
# latitudes = latitudes[northern_hemisphere_mask]  # 注意纬度可能在由正变负后还可能变正并延续一小段时间，这会导致多绘制一部分轨迹。所以我选择手动切片。 todo::修改 自动识别

latitudes = latitudes[0:140026]
longitudes = longitudes[0:140026]



# 预处理经度（转换为弧度并处理连续性）
# long_rad = np.deg2rad(longitudes - 180)  # 转换为-180~180范围
# long_rad = np.deg2rad(longitudes)  # 转换为弧度
# long_unwrapped = np.unwrap(long_rad)      # 消除跳跃
# long_deg = np.rad2deg(longitudes) + 180  # 转回0~360范围
long_deg = longitudes + 180  # 转回0~360范围
# 转换为笛卡尔坐标
r = 90 - latitudes
theta = np.deg2rad(long_deg)
x = r * np.cos(theta)
y = r * np.sin(theta)
# 绘图
plt.figure(figsize=(8,8))
ax = plt.gca() # 获取当前轴
plt.plot(x, y, linewidth=1, color='blue')
# 添加参考线
for phi in range(80, -1, -10):
    ax.add_artist(plt.Circle((0,0), 90-phi, fill=False, linestyle='--', color='gray', alpha=0.5))
    # 在纬度30和60度处添加文字标注
    if phi == 60 or phi == 30:
        radius_text = 90 - phi + 5 # 稍微向外偏移一点
        angle_rad = np.deg2rad(0) # 水平方向
        x_text = radius_text * np.cos(angle_rad)
        y_text = radius_text * np.sin(angle_rad)
        plt.text(x_text, y_text, f'{phi}°N', ha='center', va='bottom', color='gray')
for theta_deg in range(0, 360, 30):
    th = np.deg2rad(theta_deg)
    plt.plot([0, 90*np.cos(th)], [0, 90*np.sin(th)], '--', color='gray', linewidth=0.5, alpha=0.5)
    # 对经度添加标注，位置在纬度90度圈外面。
    radius_lon_text = 105 # 更靠外
    x_lon_text = radius_lon_text * np.cos(th)
    y_lon_text = radius_lon_text * np.sin(th)
    plt.text(x_lon_text, y_lon_text, f'{theta_deg}°E', ha='center', va='center', color='gray')
# 隐藏笛卡尔坐标轴。
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
# 隐藏图中的网格。
plt.grid(False)
# 在卫星轨迹的最高纬度点添加标注。
max_lat_index = np.argmax(latitudes)
max_lat_x = x[max_lat_index]
max_lat_y = y[max_lat_index]
max_latitude = latitudes[max_lat_index]
plt.plot(max_lat_x, max_lat_y, marker='o', markersize=5, color='red',label=f'Max Latitude: {max_latitude:.1f}°N') # 标记最高纬度点
plt.legend(loc='lower left', frameon=False)  # 添加图例
plt.title('Satellite Trajectory (North Pole View)')
plt.axis('equal')
plt.xlim(-120, 120) # 稍微扩大xlim和ylim以容纳经度标注
plt.ylim(-120, 120)
plt.show()