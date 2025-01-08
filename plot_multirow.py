"""
plot_with_multirow_xticks 函数可以在一条 x 轴上设置多行刻度标签，并在每行标签左侧显示对应名称。
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_with_multirow_xticks(
        x,
        rows_data,  # List[List], 每个子列表是一行要显示的数据
        rows_name=None,  # List[str], 对应每行名称，与 rows_data 长度相同
        x_label=None,
        y_data=None,  # 可选，要绘制的 y 数据（与 x 同长度）
        y_label=None,
        figsize=(6, 4)
):
    """
    在一条 x 轴上设置多行刻度标签，并在每行标签左侧显示对应名称。

    参数说明:
    -----------
    x : 1D array-like
        x 轴刻度位置, 长度 N
    rows_data : list of list (长度 M)
        每个子列表对应一行的刻度标签值，内部长度也应为 N。
        例如: rows_data = [
            [2.31, 2.32, 2.33, ...],  # UT
            [83.4, 82.0, 79.7, ...]    # MLat
        ]
    rows_name : list of str, 可选
        对应 rows_data 的每行名称，用于前缀显示；若为空则不加名称。
    x_label : str, 可选
        x 轴标题
    y_data : 1D array-like, 可选
        如果需要绘图，可传入与 x 相同长度的 y。
    y_label : str, 可选
        y 轴标题
    figsize : tuple, 默认 (6, 4)
        图像大小 (width, height) in inches

    返回值:
    -----------
    fig, ax
    """
    # 基本检查
    num_points = len(x)
    num_rows = len(rows_data)

    if rows_name is None:
        # 如果没给 rows_name，就生成空的
        rows_name = [""] * num_rows

    # 保证每行的数据长度都与 x 相同
    for row in rows_data:
        assert len(row) == num_points, "rows_data 中各列表长度应与 x 相同"

    # 生成多行标签
    # 对于第 i 个刻度，我们会将所有行的值拼到一起，如:
    #  (rows_name[0]): rows_data[0][i]
    #  (rows_name[1]): rows_data[1][i]
    #  ...
    # 并用 \n 换行。
    xtick_labels = []
    for i in range(num_points):
        # 对每一行，做一个 "名称: 值" 的小字符串
        row_strs = []
        for rname, rdata in zip(rows_name, rows_data):
            if i==0:
                prefix = f"{rname}: " if rname else ""  # 如果行名为空，就不加
                row_strs.append(f"{prefix}{rdata[i]}")
            else:
                row_strs.append(f"{rdata[i]}")
        # 用换行拼接
        label_i = "\n".join(row_strs)
        xtick_labels.append(label_i)

    # 开始画图
    fig, ax = plt.subplots(figsize=figsize)

    # 若提供了 y_data，就先画出曲线(或其他形式)
    if y_data is not None:
        ax.plot(x, y_data, marker='o', label='Example curve')

    # 设置刻度和标签
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels)

    # 设置坐标轴名称
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)

    # 若有图例，可以添加
    if y_data is not None:
        ax.legend()

    # 布局稍微紧凑一些，避免标签重叠
    fig.tight_layout()
    return fig, ax

# todo:: x数据长度过长时如何处理？

# ========== 以下为演示用示例代码 ==========

if __name__ == "__main__":
    # 假设有 6 个刻度位置
    x = np.array([0, 1, 2, 3, 4, 5])

    # 例如行1是 UT，行2是 MLat
    row1 = [2.31, 2.32, 2.35, 2.37, 2.39, 2.41]  # UT
    row2 = [83.4, 82.0, 79.7, 76.8, 73.7, 70.4]  # MLat

    # 行名
    names = ["UT", "MLat"]

    # 可选: 要画的 y (与 x 同长度)，比如随便来一个
    y_data = np.random.rand(len(x))

    fig, ax = plot_with_multirow_xticks(
        x=x,
        rows_data=[row1, row2],
        rows_name=names,
        x_label="UT / MLat (multi-row example)",
        y_data=y_data,
        y_label="Random Y"
    )

    plt.show()
