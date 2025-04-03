import numpy as np
import matplotlib  # 导入 matplotlib 库本身
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource
# from PIL import Image # Optional

# --- 添加字体配置 ---
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或 ['Microsoft YaHei'], ['Source Han Sans CN'], 等你系统有的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# --- 常量与样式设定 ---
R_E = 1.0
FIG_BG_COLOR = 'black'
AXES_BG_COLOR = '#1a1a1a'
TEXT_COLOR = 'white'
GRID_COLOR = 'gray'
EARTH_COLOR = '#4682B4'
EARTH_EDGE_COLOR = '#c0c0c0'
FIELD_LINE_COLOR = 'lime'
FIELD_LINE_ALPHA = 0.6    # 可以适当降低透明度
FIELD_LINE_WIDTH = 0.8    # 可以适当减小线宽，避免过于杂乱
START_POINT_COLOR = 'cyan'
END_POINT_COLOR = 'magenta'
POINT_SIZE = 35          # 稍微增大标记尺寸
POINT_MARKER = 'o'
SAT_TRACK_COLOR = 'yellow'
SAT_TRACK_WIDTH = 2.5    # 加粗轨道线
SAT_POS_COLOR = 'red'
SAT_POS_SIZE = 100       # 显著增大卫星标记尺寸
SAT_POS_MARKER = '*'

# !!! 新增：设置绘图的最大径向范围 (单位 R_E) !!!
# 调整这个值来控制视图范围，决定磁力线显示到多远
MAX_PLOT_RADIUS = 10.0 # 例如，只显示 10 RE 范围内的部分

# --- 1. 加载或准备你的数据 (!!! 保持不变，用你的真实数据 !!!) ---
# magnetic_field_lines = [...] # 你的磁力线数据列表
# satellite_track = ...       # 你的卫星轨道数据 (N, 3)
# satellite_current_pos_index = ... # 卫星位置索引

# 示例占位数据 (你需要替换！)
num_lines = 50 # 增加磁力线数量以模拟更复杂场景
magnetic_field_lines = []
for i in range(num_lines):
    # 模拟一些延伸较远的线
    if i % 5 == 0:
        theta = np.linspace(np.pi / 6, 5 * np.pi / 6, 150)
        r = 1.5 + 15 * np.sin(theta)**2 # 最高可达 16.5 RE
    else:
        theta = np.linspace(np.pi / 4, 3 * np.pi / 4, 100)
        r = 1.5 + 3 * np.sin(theta)**2 # 最高约 4.5 RE
    phi = (i / num_lines) * 2 * np.pi
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    line_data = np.vstack((x, y, z)).T
    magnetic_field_lines.append(line_data)

theta_sat = np.linspace(0, 2 * np.pi, 200)
a_sat = 4 * R_E
e_sat = 0.1
x_sat = a_sat * (np.cos(theta_sat) - e_sat)
y_sat = a_sat * np.sqrt(1 - e_sat**2) * np.sin(theta_sat)
z_sat = 0.5 * R_E * np.sin(theta_sat*2)
satellite_track = np.vstack((x_sat, y_sat, z_sat)).T
satellite_current_pos_index = 50

# --- (更新 get_cmap 的用法，消除 DeprecationWarning) ---
def plot_earth_shaded(ax, radius=R_E, color=EARTH_COLOR, edge_color=None, texture_path=None, light_source_angles=(315, 45)):
    u = np.linspace(0, 2 * np.pi, 120)
    v = np.linspace(0, np.pi, 60)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))

    ls = LightSource(azdeg=light_source_angles[0], altdeg=light_source_angles[1])
    from matplotlib.colors import ListedColormap, to_rgba
    ls = LightSource(azdeg=light_source_angles[0], altdeg=light_source_angles[1])
    from matplotlib.colors import ListedColormap, to_rgba
    # base_cmap = plt.cm.get_cmap('Blues_r', 256)
    base_cmap = matplotlib.colormaps['Blues_r']  # 新用法
    # Ensure base color alpha is 1 for shading calculation
    base_color_rgba = list(to_rgba(color))
    base_color_rgba[3] = 1.0
    shaded_cmap = ListedColormap(base_cmap(np.linspace(0.2, 1, 256)) * np.array(base_color_rgba))

    rgb = ls.shade(z, cmap=shaded_cmap, vert_exag=0.1, blend_mode='soft')

    surf = ax.plot_surface(x, y, z, rstride=2, cstride=2, facecolors=rgb,
                           linewidth=0.1, antialiased=True, alpha=0.9, zorder=1) # 地球 zorder=1
    if edge_color:
         surf.set_edgecolor(edge_color)
         surf.set_linewidth(0.5)

# --- 3. 主绘图程序 ---
print("开始绘图...")
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
fig.patch.set_facecolor(FIG_BG_COLOR)
ax.set_facecolor(AXES_BG_COLOR)

# 绘制地球
plot_earth_shaded(ax, radius=R_E, color=EARTH_COLOR)
print("地球绘制完成.")

# 绘制磁力线 (进行数据截断)
print(f"绘制 {len(magnetic_field_lines)} 条磁力线 (截断在 {MAX_PLOT_RADIUS} RE)...")
all_start_points = []
all_end_points = []
num_plotted_lines = 0
for i, line_data in enumerate(magnetic_field_lines):
    if line_data.shape[0] == 3 and line_data.shape[1] > 3:
        line_data = line_data.T
    elif line_data.shape[1] != 3:
        continue
    if line_data.shape[0] < 2:
        continue

    # !!! 数据截断 !!!
    r = np.sqrt(np.sum(line_data**2, axis=1))
    mask = r <= MAX_PLOT_RADIUS
    plot_data = line_data[mask]

    # 只有截断后仍有点才绘制
    if plot_data.shape[0] >= 2:
        ax.plot(plot_data[:, 0], plot_data[:, 1], plot_data[:, 2],
                color=FIELD_LINE_COLOR, alpha=FIELD_LINE_ALPHA, linewidth=FIELD_LINE_WIDTH,
                zorder=2) # 磁力线 zorder=2

        # 记录截断后的起点和终点
        all_start_points.append(plot_data[0, :])
        all_end_points.append(plot_data[-1, :])
        num_plotted_lines += 1
    elif plot_data.shape[0] == 1: # 如果截断后只剩一个点，也标记一下
        all_start_points.append(plot_data[0, :]) # 视为起点
        num_plotted_lines += 1


# 统一绘制起点和终点
if all_start_points:
    starts = np.array(all_start_points)
    ax.scatter(starts[:, 0], starts[:, 1], starts[:, 2],
               color=START_POINT_COLOR, marker=POINT_MARKER, s=POINT_SIZE,
               label=f'磁力线起点/截断点 ({num_plotted_lines}条)', zorder=4) # zorder=4 靠前
if all_end_points:
    ends = np.array(all_end_points)
    # 如果起点和终点标记相同，只加一个label即可
    ax.scatter(ends[:, 0], ends[:, 1], ends[:, 2],
               color=END_POINT_COLOR, marker=POINT_MARKER, s=POINT_SIZE,
               zorder=4)
print("磁力线绘制完成.")

# 绘制卫星轨道 (进行数据截断)
print("绘制卫星轨道...")
if satellite_track.shape[0] > 1:
    if satellite_track.shape[0] == 3 and satellite_track.shape[1] > 3:
        satellite_track = satellite_track.T
    elif satellite_track.shape[1] != 3:
         print(f"  警告: 卫星轨道数据形状 {satellite_track.shape} 不符合要求，跳过。")
    else:
        # !!! 数据截断 !!!
        r_sat = np.sqrt(np.sum(satellite_track**2, axis=1))
        mask_sat = r_sat <= MAX_PLOT_RADIUS
        plot_sat_track = satellite_track[mask_sat]

        if plot_sat_track.shape[0] >= 2:
            ax.plot(plot_sat_track[:, 0], plot_sat_track[:, 1], plot_sat_track[:, 2],
                    color=SAT_TRACK_COLOR, linewidth=SAT_TRACK_WIDTH, label='卫星轨道',
                    zorder=3) # 卫星轨道 zorder=3

        # 标记卫星当前位置 (如果该位置在绘图范围内)
        if 0 <= satellite_current_pos_index < satellite_track.shape[0]:
            sat_pos = satellite_track[satellite_current_pos_index, :]
            r_sat_pos = np.sqrt(np.sum(sat_pos**2))
            if r_sat_pos <= MAX_PLOT_RADIUS: # 只绘制范围内的标记
                ax.scatter(sat_pos[0], sat_pos[1], sat_pos[2],
                           color=SAT_POS_COLOR, marker=SAT_POS_MARKER, s=SAT_POS_SIZE,
                           label='卫星位置示意', zorder=5) # 卫星标记 zorder=5 (最高)
            else:
                 print(f"  信息: 卫星当前位置 ({r_sat_pos:.1f} RE) 超出绘图范围 {MAX_PLOT_RADIUS} RE，未标记。")
        else:
            print("  警告: 卫星当前位置索引无效。")
else:
    print("  警告: 卫星轨道数据点不足。")
print("卫星轨道绘制完成.")


# --- 4. 精调外观与设置 ---

# 设置坐标轴标签
ax.set_xlabel('X [$R_E$]', color=TEXT_COLOR, fontsize=12)
ax.set_ylabel('Y [$R_E$]', color=TEXT_COLOR, fontsize=12)
ax.set_zlabel('Z [$R_E$]', color=TEXT_COLOR, fontsize=12)

# 设置标题
ax.set_title(f'地球磁场线与卫星轨道 (视图范围: {MAX_PLOT_RADIUS} $R_E$)', color=TEXT_COLOR, fontsize=14, weight='bold')

# 设置坐标轴范围 (基于 MAX_PLOT_RADIUS)
plot_limit = MAX_PLOT_RADIUS * 1.05 # 留一点边距
ax.set_xlim([-plot_limit, plot_limit])
ax.set_ylim([-plot_limit, plot_limit])
ax.set_zlim([-plot_limit, plot_limit])

# 设置等比例坐标轴
limits = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
ax.set_box_aspect(np.ptp(limits, axis=1))

# 设置刻度、面板、网格线等 (保持不变)
ax.tick_params(axis='x', colors=TEXT_COLOR)
ax.tick_params(axis='y', colors=TEXT_COLOR)
ax.tick_params(axis='z', colors=TEXT_COLOR)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.line.set_color(TEXT_COLOR)
ax.yaxis.line.set_color(TEXT_COLOR)
ax.zaxis.line.set_color(TEXT_COLOR)
ax.grid(True, color=GRID_COLOR, linestyle=':', linewidth=0.5, alpha=0.5)

# 添加图例
legend = ax.legend(facecolor=AXES_BG_COLOR, edgecolor=TEXT_COLOR, framealpha=0.8, loc='upper left', bbox_to_anchor=(0.01, 0.99)) # 调整图例位置
plt.setp(legend.get_texts(), color=TEXT_COLOR)

# 调整视角
ax.view_init(elev=25., azim=-110)

plt.tight_layout()

# --- 5. 保存图像 ---
output_filename_png = f'magnetosphere_plot_limited_{MAX_PLOT_RADIUS}RE.png'
plt.savefig(output_filename_png, dpi=300, facecolor=FIG_BG_COLOR, bbox_inches='tight')
print(f"图形已保存为 {output_filename_png}")
# plt.show()