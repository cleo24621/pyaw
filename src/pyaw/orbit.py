import numpy as np
from cartopy import crs as ccrs
import cartopy.feature as cfeature # Moved import here for clarity
from matplotlib import pyplot as plt, path as mpath
from matplotlib.axes import Axes
from numpy.typing import NDArray


# todo: simplify

def get_nor_sou_split_indices_swarm_dmsp(latitudes: NDArray):
    """
    get the indices that split array into northern and southern.
    todo: 也许需要添加处理纬度等于0时属于北半球或南半球的情况（会有吗？（极端情况：0.00001））
    Args:
        latitudes: pattern is [south,north], or [south,north,south(few)]

    """
    neg_indices = np.where(latitudes < 0)[0]
    if not neg_indices.size:
        return (0, len(latitudes)), (len(latitudes), len(latitudes))

    start_south = neg_indices[0]
    pos_indices = np.where(latitudes[start_south:] >= 0)[0]
    end_south = start_south + pos_indices[0] if pos_indices.size else len(latitudes)

    return (0, start_south), (start_south, end_south)


class OrbitZh1:
    def __init__(self, file_name):
        """

        Args:
            file_name: support the efd 2a data filename
        """
        self.file_name = file_name
        self.orbit_number, self.indicator, self.start_time, self.end_time = (
            self._get_orbitnumber_indicator_st_et()
        )

    def _get_orbitnumber_indicator_st_et(self):
        """
        因为不同2级产品的命名格式是固定的，所以当前方法适用于所有2a级产品的文件名（参考文档）。
        1: ascending (south to north)
        0: descending (north to south)
        """
        parts = self.file_name.split("_")
        part = parts[6]
        assert part[-1] in ["0", "1"]
        start_time = parts[7] + "_" + parts[8]
        end_time = parts[9] + "_" + parts[10]
        return parts[6][:-1], parts[6][-1], start_time, end_time

    def get_nor_sou_split_indices(self, latitudes: NDArray):
        """

        Returns:

        """
        # 对于zh1而言，indicator="0" or indicator="1"
        assert self.indicator in ["1", "0"]
        if all(latitudes > 0):
            return (0, len(latitudes)), (len(latitudes), len(latitudes))
        elif all(latitudes < 0):
            return (len(latitudes), len(latitudes)), (0, len(latitudes))
        elif self.indicator == "1":
            start_north = np.where(latitudes > 0)[0][0]
            return (start_north, None), (0, start_north)  # north, south slice
        else:
            start_south = np.where(latitudes < 0)[0][0]
            return (0, start_south), (start_south, None)  # north, south slice


def orbit_hemisphere_projection(
    lons: NDArray,
    lats: NDArray,
    proj_method: str,
    satellite: str = None,
    central_longitude: float = 0,
    figsize: tuple[float, float] = (12, 12),
    lon_lat_ext: tuple[float, float, float, float] = None,
    if_cfeature: bool = False,
    if_title: bool = True,
    title: str = None,
    ax: Axes = None,
):
    """

    Args:
        lons: hemisphere lons (half orbit)
        lats: ~
        satellite:
        proj_method: one of ["NorthPolarStereo","SouthPolarStereo"]
        central_longitude:
        figsize:
        lon_lat_ext: 经度和纬度的范围
        if_cfeature: 是否显示地理信息（海岸线等）
        if_title: 是否显示标题
        title: 标题
        ax: Axes对象，为南北同时绘制做准备

    Returns:

    """
    assert proj_method in ["NorthPolarStereo", "SouthPolarStereo"]
    # 投影配置
    proj_dict = {
        "NorthPolarStereo": ccrs.NorthPolarStereo,
        "SouthPolarStereo": ccrs.SouthPolarStereo,
    }
    proj = proj_dict[proj_method](central_longitude=central_longitude)
    geodetic = ccrs.PlateCarree()

    # 设置默认的extent,标题,ylocs
    if proj_method == "NorthPolarStereo":
        default_ext = (-180, 180, 0, 90)
        default_title = "North Hemisphere Map using NorthPolarStereo Projection"
        ylocs = np.arange(0, 91, 15)

    else:
        default_ext = (-180, 180, -90, 0)
        default_title = "South Hemisphere Map using SouthPolarStereo Projection"
        ylocs = np.arange(-90, 0, 15)

    current_ext = lon_lat_ext if lon_lat_ext is not None else default_ext
    current_title = title if title is not None else default_title

    # ax为None时创建图像，不为None时使用其原来对应的图像
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": proj})
    else:
        fig = ax.figure

    # set border
    ax.set_extent(list(current_ext), crs=geodetic)
    # create circle border
    theta = np.linspace(0, 2 * np.pi, 100)
    circle = mpath.Path(np.column_stack([np.sin(theta), np.cos(theta)]) * 0.45 + 0.5)
    ax.set_boundary(circle, transform=ax.transAxes)
    # 地理信息
    if if_cfeature:
        import cartopy.feature as cfeature

        ax.add_feature(cfeature.LAND.with_scale("50m"), alpha=0.6)
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), alpha=0.4)
        ax.coastlines(resolution="50m", linewidth=0.5)

    # 添加网格线
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        xlocs=np.arange(-180, 181, 45),
        ylocs=ylocs,
        xpadding=15,
        ypadding=15,
    )
    gl.top_labels, gl.right_labels, gl.rotate_labels = False, False, False

    # plot
    ax.plot(
        *proj.transform_points(geodetic, lons, lats)[:, :2].T,
        "b-",
        lw=1.5,
        label="Satellite orbit" if satellite is None else f"{satellite} Orbit",
    )
    ax.legend(loc="upper right")
    # show title or not
    if if_title:
        ax.set_title(current_title, fontsize=12, pad=18)
    return fig, ax


def orbit_hemispheres_projection(
    lons_nor: NDArray,
    lats_nor: NDArray,
    lons_sou: NDArray,
    lats_sou: NDArray,
    satellite: str = None,
    central_longitude: float = 0,
    figsize: tuple[float, float] = (24, 12),
    figsize_sub=(12, 12),
    lon_lat_ext: tuple[float, float, float, float] = None,
    if_cfeature: bool = False,
    if_subtitle: bool = True,
    subtitle: str = None,
):
    """

    Args:
        if_subtitle: 子图是否显示标题
        subtitle: 子图的标题
        figsize_sub: 子图的大小
        lons_nor: north lons (half orbit)
        lats_nor: ~
        lons_sou:
        lats_sou:
        satellite:
        central_longitude:
        figsize:
        lon_lat_ext:
        if_cfeature: 是否显示地理信息（海岸线等）

    Returns:

    """
    # 创建图形和子图
    fig = plt.figure(figsize=figsize)
    # nor
    ax1 = fig.add_subplot(
        1, 2, 1, projection=ccrs.NorthPolarStereo(central_longitude=central_longitude)
    )
    orbit_hemisphere_projection(
        lons_nor,
        lats_nor,
        "NorthPolarStereo",
        satellite,
        central_longitude,
        figsize_sub,
        lon_lat_ext,
        if_cfeature,
        if_subtitle,
        subtitle,
        ax=ax1,
    )
    # sou
    ax2 = fig.add_subplot(
        1, 2, 2, projection=ccrs.SouthPolarStereo(central_longitude=central_longitude)
    )
    orbit_hemisphere_projection(
        lons_sou,
        lats_sou,
        "SouthPolarStereo",
        satellite,
        central_longitude,
        figsize_sub,
        lon_lat_ext,
        if_cfeature,
        if_subtitle,
        subtitle,
        ax=ax2,
    )
    return fig, ax1, ax2


def orbits_hemisphere_projection(
    lons_list: list[NDArray],
    lats_list: list[NDArray],
    proj_method: str,
    central_longitude: float = 0,
    figsize: tuple[float, float] = (12, 12),
    lon_lat_ext: tuple[float, float, float, float] = None,
    if_cfeature: bool = False,
    if_title: bool = True,
    title: str = None,
    ax: Axes = None,
):
    """
    绘制多轨半球投影
    Args:
        ax:
        title:
        proj_method:
        lons_list: the element of the list is 1 hemisphere lons. And the length of the list should be equal to lats.
        lats_list: ~
        central_longitude:
        figsize:
        lon_lat_ext:
        if_cfeature: 是否显示地理信息（海岸线等）
        if_title: 是否显示标题

    Returns:

    """
    assert proj_method in ["NorthPolarStereo", "SouthPolarStereo"]
    # 投影配置
    proj_dict = {
        "NorthPolarStereo": ccrs.NorthPolarStereo,
        "SouthPolarStereo": ccrs.SouthPolarStereo,
    }
    proj = proj_dict[proj_method](central_longitude=central_longitude)
    geodetic = ccrs.PlateCarree()

    # 设置默认的extent,标题,ylocs
    if proj_method == "NorthPolarStereo":
        default_ext = (-180, 180, 0, 90)
        default_title = "North Hemisphere Map using NorthPolarStereo Projection"
        ylocs = np.arange(0, 91, 15)
        # for zh1 limit label
        # ylocs = np.append(ylocs,65)
        # ylocs = np.sort(ylocs)  # 确保纬度按升序排列

    else:
        default_ext = (-180, 180, -90, 0)
        default_title = "South Hemisphere Map using SouthPolarStereo Projection"
        ylocs = np.arange(-90, 0, 15)
        # for zh1 limit label
        # ylocs = np.append(ylocs,-65)
        # ylocs = np.sort(ylocs)  # 确保纬度按升序排列

    current_ext = lon_lat_ext if lon_lat_ext is not None else default_ext
    current_title = title if title is not None else default_title

    # ax为None时创建图像，不为None时使用其原来对应的图像
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": proj})
    else:
        fig = ax.figure

    # set border
    ax.set_extent(list(current_ext), crs=geodetic)
    # create circle border
    theta = np.linspace(0, 2 * np.pi, 100)
    circle = mpath.Path(np.column_stack([np.sin(theta), np.cos(theta)]) * 0.45 + 0.5)
    ax.set_boundary(circle, transform=ax.transAxes)
    # 地理信息
    if if_cfeature:
        import cartopy.feature as cfeature

        ax.add_feature(cfeature.LAND.with_scale("50m"), alpha=0.6)
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), alpha=0.4)
        ax.coastlines(resolution="50m", linewidth=0.5)

    # 添加网格线
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        xlocs=np.arange(-180, 181, 45),
        ylocs=ylocs,
        xpadding=15,
        ypadding=15,
    )
    gl.top_labels, gl.right_labels, gl.rotate_labels = False, False, False
    # modify labels
    # gl.xlabels_top = False  # 禁用顶部经度标签
    # gl.ylabels_right = False  # 禁用右侧纬度标签
    # gl.ylabels_left = True  # 显式启用左侧纬度标签

    # plot
    for lons, lats in zip(lons_list, lats_list):
        ax.plot(
            *proj.transform_points(geodetic, lons, lats)[:, :2].T,
            "b-",
            lw=1.5,
            # label="Satellite orbit" if satellite is None else f"{satellite} Orbit", (no label because of will have too many labels)
        )
    if if_title:
        ax.set_title(current_title, fontsize=12, pad=18)
    return fig, ax


def orbits_hemispheres_projection(
    lons_nor_list: list[NDArray],
    lats_nor_list: list[NDArray],
    lons_sou_list: list[NDArray],
    lats_sou_list: list[NDArray],
    central_longitude: float = 0,
    figsize: tuple[float, float] = (24, 12),
    figsize_sub: tuple[float, float] = (12, 12),
    lon_lat_ext: tuple[float, float, float, float] = None,
    if_cfeature: bool = False,
    if_subtitle: bool = True,
    subtitle: str = None,
):
    """

    Args:
        subtitle:
        figsize_sub:
        lons_nor_list: the element of the list is north hemisphere lons. And the length of the list should be equal to other lists (lats_nor,lons_sou,lats_sou).
        lats_nor_list: ~
        lats_sou_list:
        lons_sou_list:
        central_longitude:
        figsize:
        lon_lat_ext:
        if_cfeature: 是否显示地理信息（海岸线等）
        if_subtitle: 是否显示标题

    Returns:

    """
    # 创建图形和子图
    fig = plt.figure(figsize=figsize)
    # nor
    ax1 = fig.add_subplot(
        1, 2, 1, projection=ccrs.NorthPolarStereo(central_longitude=central_longitude)
    )
    orbits_hemisphere_projection(
        lons_nor_list,
        lats_nor_list,
        "NorthPolarStereo",
        central_longitude,
        figsize_sub,
        lon_lat_ext,
        if_cfeature,
        if_subtitle,
        subtitle,
        ax=ax1,
    )
    # sou
    ax2 = fig.add_subplot(
        1, 2, 2, projection=ccrs.SouthPolarStereo(central_longitude=central_longitude)
    )
    orbits_hemisphere_projection(
        lons_sou_list,
        lats_sou_list,
        "SouthPolarStereo",
        central_longitude,
        figsize_sub,
        lon_lat_ext,
        if_cfeature,
        if_subtitle,
        subtitle,
        ax=ax2,
    )
    return fig, ax1, ax2


def orbit_hemisphere_with_vector_projection(
    lons,
    lats,
    vector_east,
    vector_north,
    proj_method,
    central_longitude=0,
    zorder: float = 2,
):
    fig, ax = orbit_hemisphere_projection(lons=lons, lats=lats, proj_method=proj_method)
    proj_dict = {
        "NorthPolarStereo": ccrs.NorthPolarStereo,
        "SouthPolarStereo": ccrs.SouthPolarStereo,
    }
    proj = proj_dict[proj_method](central_longitude=central_longitude)
    geodetic = ccrs.PlateCarree()

    u_proj, v_proj = proj.transform_vectors(
        geodetic, lons, lats, vector_east, vector_north
    )
    q = ax.quiver(
        lons,
        lats,
        u_proj,
        v_proj,
        scale=35,  # 增大scale值以适应更大范围
        width=0.0025,  # 更细的箭头杆
        headwidth=0,
        headlength=0,
        headaxislength=0,  # 箭头轴长度（可选）(default=4.5，设置不同值的影响查看示例)（此绘制最好设置为0）
        color="crimson",
        transform=geodetic,
        zorder=zorder,
    )
    ax.quiverkey(
        q,
        X=0.82,
        Y=0.12,
        U=0.3,
        label=f"Normalized resultant vector of the eastward and northward components",
        labelpos="E",  # 标签在箭头右侧
        coordinates="axes",
    )
    return fig, ax


def orbits_hemisphere_with_vector_projection(
    lons_list: list[NDArray],
    lats_list: list[NDArray],
    vector_east_list,
    vector_north_list,
    proj_method: str,
    central_longitude: float = 0,
    figsize: tuple[float, float] = (12, 12),
    lon_lat_ext: tuple[float, float, float, float] = None,
    if_cfeature: bool = False,
    if_title: bool = True,
    title: str = None,
    ax: Axes = None,
    step=None,
):
    """
    绘制多轨半球投影
    Args:
        ax:
        title:
        proj_method:
        lons_list: the element of the list is 1 hemisphere lons. And the length of the list should be equal to lats.
        lats_list: ~
        central_longitude:
        figsize:
        lon_lat_ext:
        if_cfeature: 是否显示地理信息（海岸线等）
        if_title: 是否显示标题

    Returns:

    """
    assert proj_method in ["NorthPolarStereo", "SouthPolarStereo"]
    # 投影配置
    proj_dict = {
        "NorthPolarStereo": ccrs.NorthPolarStereo,
        "SouthPolarStereo": ccrs.SouthPolarStereo,
    }
    proj = proj_dict[proj_method](central_longitude=central_longitude)
    geodetic = ccrs.PlateCarree()

    # 设置默认的extent,标题,ylocs
    if proj_method == "NorthPolarStereo":
        default_ext = (-180, 180, 0, 90)
        default_title = "North Hemisphere Map using NorthPolarStereo Projection"
        ylocs = np.arange(0, 91, 15)
        # for zh1 limit label
        # ylocs = np.append(ylocs,65)
        # ylocs = np.sort(ylocs)  # 确保纬度按升序排列

    else:
        default_ext = (-180, 180, -90, 0)
        default_title = "South Hemisphere Map using SouthPolarStereo Projection"
        ylocs = np.arange(-90, 0, 15)
        # for zh1 limit label
        # ylocs = np.append(ylocs,-65)
        # ylocs = np.sort(ylocs)  # 确保纬度按升序排列

    current_ext = lon_lat_ext if lon_lat_ext is not None else default_ext
    current_title = title if title is not None else default_title

    # ax为None时创建图像，不为None时使用其原来对应的图像
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": proj})
    else:
        fig = ax.figure

    # set border
    ax.set_extent(list(current_ext), crs=geodetic)
    # create circle border
    theta = np.linspace(0, 2 * np.pi, 100)
    circle = mpath.Path(np.column_stack([np.sin(theta), np.cos(theta)]) * 0.45 + 0.5)
    ax.set_boundary(circle, transform=ax.transAxes)
    # 地理信息
    if if_cfeature:
        import cartopy.feature as cfeature

        ax.add_feature(cfeature.LAND.with_scale("50m"), alpha=0.6)
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), alpha=0.4)
        ax.coastlines(resolution="50m", linewidth=0.5)

    # 添加网格线
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        xlocs=np.arange(-180, 181, 45),
        ylocs=ylocs,
        xpadding=15,
        ypadding=15,
    )
    gl.top_labels, gl.right_labels, gl.rotate_labels = False, False, False
    # modify labels
    # gl.xlabels_top = False  # 禁用顶部经度标签
    # gl.ylabels_right = False  # 禁用右侧纬度标签
    # gl.ylabels_left = True  # 显式启用左侧纬度标签

    # plot
    for lons, lats, vector_east, vector_north in zip(
        lons_list, lats_list, vector_east_list, vector_north_list
    ):
        # orbit projection
        ax.plot(
            *proj.transform_points(geodetic, lons[::step], lats[::step])[:, :2].T,
            "b-",
            lw=1.5,
            zorder=2,
            # label="Satellite orbit" if satellite is None else f"{satellite} Orbit", (no label because of will have too many labels)
        )
        # vector projection
        # 转换矢量并绘制
        u_proj, v_proj = proj.transform_vectors(
            geodetic,
            lons[::step],
            lats[::step],
            vector_east[::step],
            vector_north[::step],
        )
        q = ax.quiver(
            lons[::step],
            lats[::step],
            u_proj,
            v_proj,
            scale=35,  # 增大scale值以适应更大范围
            width=0.0025,  # 更细的箭头杆
            headwidth=0,
            headlength=0,
            headaxislength=0,  # 箭头轴长度（可选）(default=4.5，设置不同值的影响查看示例)（此绘制最好设置为0）
            color="crimson",
            transform=geodetic,
            zorder=4,
        )
        ax.quiverkey(
            q,
            X=0.82,
            Y=0.12,
            U=0.3,
            label=f"Normalized Measured Minus Model Magnetic Field (nT)",
            labelpos="E",  # 标签在箭头右侧
            coordinates="axes",
        )

    if if_title:
        ax.set_title(current_title, fontsize=12, pad=18)
    return fig, ax


def orbits_hemispheres_with_vector_projection():
    pass


def orbits_hemisphere_with_one_vector_projection(
    lons_list: list[np.ndarray],
    lats_list: list[np.ndarray],
    vector_component_list,
    proj_method: str,
    central_longitude: float = 0,
    figsize: tuple[float, float] = (12, 12),
    lon_lat_ext: tuple[float, float, float, float] = None,
    if_cfeature: bool = False,
    if_title: bool = True,
    title: str = None,
    ax: plt.Axes = None,
    step: int = None,
):
    assert proj_method in ["NorthPolarStereo", "SouthPolarStereo"]
    # 投影配置
    proj_dict = {
        "NorthPolarStereo": ccrs.NorthPolarStereo,
        "SouthPolarStereo": ccrs.SouthPolarStereo,
    }
    proj = proj_dict[proj_method](central_longitude=central_longitude)
    geodetic = ccrs.PlateCarree()

    # 设置默认的extent,标题,ylocs
    if proj_method == "NorthPolarStereo":
        default_ext = (-180, 180, 0, 90)
        default_title = "North Hemisphere Map using NorthPolarStereo Projection"
        ylocs = np.arange(0, 91, 15)
        # for zh1 limit label
        # ylocs = np.append(ylocs,65)
        # ylocs = np.sort(ylocs)  # 确保纬度按升序排列

    else:
        default_ext = (-180, 180, -90, 0)
        default_title = "South Hemisphere Map using SouthPolarStereo Projection"
        ylocs = np.arange(-90, 0, 15)
        # for zh1 limit label
        # ylocs = np.append(ylocs,-65)
        # ylocs = np.sort(ylocs)  # 确保纬度按升序排列

    current_ext = lon_lat_ext if lon_lat_ext is not None else default_ext
    current_title = title if title is not None else default_title

    # ax为None时创建图像，不为None时使用其原来对应的图像
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": proj})
    else:
        fig = ax.figure

    # set border
    ax.set_extent(list(current_ext), crs=geodetic)
    # create circle border
    theta = np.linspace(0, 2 * np.pi, 100)
    circle = mpath.Path(np.column_stack([np.sin(theta), np.cos(theta)]) * 0.45 + 0.5)
    ax.set_boundary(circle, transform=ax.transAxes)
    # 地理信息
    if if_cfeature:
        import cartopy.feature as cfeature

        ax.add_feature(cfeature.LAND.with_scale("50m"), alpha=0.6)
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), alpha=0.4)
        ax.coastlines(resolution="50m", linewidth=0.5)

    # 添加网格线
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        xlocs=np.arange(-180, 181, 45),
        ylocs=ylocs,
        xpadding=15,
        ypadding=15,
    )
    gl.top_labels, gl.right_labels, gl.rotate_labels = False, False, False

    first = True  # 控制quiverkey仅显示一次
    for lons, lats, vector_component in zip(lons_list, lats_list, vector_component_list):
        lons_step = lons[::step]
        lats_step = lats[::step]
        comp_step = vector_component[::step]

        # 转换到投影坐标
        proj_points = proj.transform_points(geodetic, lons_step, lats_step)
        x = proj_points[:, 0]
        y = proj_points[:, 1]

        # 绘制轨迹线
        ax.plot(x, y, 'b-', lw=1.5, zorder=2)

        # 计算矢量方向（垂直于轨迹）
        if len(x) < 2:
            continue
        dx = np.diff(x)
        dy = np.diff(y)
        perp_dx = -dy
        perp_dy = dx

        # 归一化垂直方向
        lengths = np.hypot(perp_dx, perp_dy)
        mask = lengths != 0
        perp_dx[mask] /= lengths[mask]
        perp_dy[mask] /= lengths[mask]

        components = comp_step[:-1]  # len(perp_dx) - len(comp_step) = 1
        max_component = np.max(np.abs(components))
        if max_component == 0:
            continue

        # 缩放矢量长度
        scale_factor = 2 / max_component
        scaled = components * scale_factor
        u = perp_dx * scaled
        v = perp_dy * scaled

        # 绘制箭头
        q = ax.quiver(
            x[:-1], y[:-1],
            u, v,
            scale=35,
            width=0.0025,
            headwidth=0,
            headlength=0,
            headaxislength=0,
            color='crimson',
            transform=proj,
            zorder=4
        )

        # 添加quiverkey（仅第一次绘制）
        if first:
            ax.quiverkey(
                q, X=0.82, Y=0.12, U=0.5,
                label=f'Normalized Field (nT)',
                labelpos='E',
                coordinates='axes'
            )
            first = False
    if if_title:
        ax.set_title(current_title, fontsize=12, pad=18)
    return fig, ax

def orbits_hemisphere_with_one_vector_projection_modified(
    lons_list: list[np.ndarray],
    lats_list: list[np.ndarray],
    vector_component_list,
    proj_method: str,
    central_longitude: float = 0,
    figsize: tuple[float, float] = (12, 12),
    lon_lat_ext: tuple[float, float, float, float] = None,
    if_cfeature: bool = False,
    if_title: bool = True,
    title: str = None,
    ax: plt.Axes = None,
    step: int = None,
    vector_scale_numerator: float = 2.0, # New parameter for scale factor numerator
):
    """
    Plots satellite orbits and associated vector components perpendicular to the track
    on a North/South Polar Stereographic projection.

    Args:
        lons_list (list[np.ndarray]): List of longitude arrays for each orbit.
        lats_list (list[np.ndarray]): List of latitude arrays for each orbit.
        vector_component_list (list[np.ndarray]): List of scalar component arrays
                                                 for vectors perpendicular to the track.
        proj_method (str): Projection method, either "NorthPolarStereo" or "SouthPolarStereo".
        central_longitude (float, optional): Central longitude for the projection. Defaults to 0.
        figsize (tuple[float, float], optional): Figure size. Defaults to (12, 12).
        lon_lat_ext (tuple[float, float, float, float], optional): Map extent
                                                               (lon_min, lon_max, lat_min, lat_max).
                                                               Defaults based on proj_method.
        if_cfeature (bool, optional): Whether to add Cartopy land/ocean features. Defaults to False.
        if_title (bool, optional): Whether to add a title to the plot. Defaults to True.
        title (str, optional): Custom title string. Defaults based on proj_method.
        ax (plt.Axes, optional): Existing Axes object to plot on. If None, creates a new figure and axes. Defaults to None.
        step (int, optional): Step size for subsampling orbit points and vectors. Defaults to None (no subsampling).
        vector_scale_numerator (float, optional): Numerator used for scaling vector lengths
                                                 (scale = numerator / max_abs_component). Defaults to 2.0.
    Returns:
        tuple[plt.Figure, plt.Axes]: The figure and axes objects.
    """
    assert proj_method in ["NorthPolarStereo", "SouthPolarStereo"], \
        "proj_method must be either 'NorthPolarStereo' or 'SouthPolarStereo'"

    # Projection configuration
    proj_dict = {
        "NorthPolarStereo": ccrs.NorthPolarStereo,
        "SouthPolarStereo": ccrs.SouthPolarStereo,
    }
    proj = proj_dict[proj_method](central_longitude=central_longitude)
    geodetic = ccrs.PlateCarree()

    # Set default extent, title, and ylocs based on projection
    if proj_method == "NorthPolarStereo":
        default_ext = (-180, 180, 0, 90)
        default_title = "North Hemisphere Map using NorthPolarStereo Projection"
        ylocs = np.arange(0, 91, 15)
    else: # SouthPolarStereo
        default_ext = (-180, 180, -90, 0)
        default_title = "South Hemisphere Map using SouthPolarStereo Projection"
        ylocs = np.arange(-90, 0, 15)

    current_ext = lon_lat_ext if lon_lat_ext is not None else default_ext
    current_title = title if title is not None else default_title

    # Create figure/axes if 'ax' is not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": proj})
    else:
        fig = ax.figure # Use the figure associated with the provided axes

    # Set map extent
    ax.set_extent(list(current_ext), crs=geodetic)

    # Create circular boundary for the polar plot
    # Adjust radius calculation for potentially non-square figures if needed,
    # but 0.45 is often visually acceptable for square-ish polar plots.
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

    # Add geographic features if requested
    if if_cfeature:
        ax.add_feature(cfeature.LAND.with_scale("50m"), alpha=0.6, zorder=0)
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), alpha=0.4, zorder=0)
        ax.coastlines(resolution="50m", linewidth=0.5, color='gray', zorder=1)

    # Add gridlines
    gl = ax.gridlines(
        crs=geodetic, # Specify CRS for gridlines explicitly
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.7,
        linestyle='--',
        xlocs=np.arange(-180, 181, 45),
        ylocs=ylocs,
        # Consider label padding adjustments if labels overlap border
        # xpadding=15,
        # ypadding=15,
    )
    gl.top_labels, gl.right_labels, gl.rotate_labels = False, False, False
    # Prettier longitude labels (optional)
    # gl.xformatter = LONGITUDE_FORMATTER
    # gl.yformatter = LATITUDE_FORMATTER

    first_quiver = True  # Control quiverkey display (only once)
    for lons, lats, vector_component in zip(lons_list, lats_list, vector_component_list):
        # Apply step for subsampling if specified
        if step is not None and step > 1:
             lons_step = lons[::step]
             lats_step = lats[::step]
             comp_step = vector_component[::step]
        else:
             lons_step = lons
             lats_step = lats
             comp_step = vector_component

        # Transform coordinates to map projection
        # Note: Ensure lons/lats are within standard ranges (-180 to 180, -90 to 90)
        # before transformation if they might not be.
        try:
            proj_points = proj.transform_points(geodetic, lons_step, lats_step)
            x = proj_points[:, 0]
            y = proj_points[:, 1]
        except Exception as e:
            print(f"Warning: Coordinate transformation failed for an orbit segment: {e}")
            continue # Skip this orbit segment if transformation fails

        # Plot trajectory line
        # --- MODIFICATION: Changed color to 'darkgray' ---
        ax.plot(x, y, color='darkgray', linestyle='-', lw=1.5, zorder=2, transform=proj)

        # Calculate vector direction (perpendicular to trajectory)
        if len(x) < 2:
            # Need at least two points to calculate direction
            continue
        dx = np.diff(x)
        dy = np.diff(y)
        perp_dx = -dy
        perp_dy = dx

        # Normalize perpendicular direction vectors
        lengths = np.hypot(perp_dx, perp_dy)
        # Avoid division by zero for segments with zero length (e.g., repeated points)
        valid_mask = lengths != 0
        if not np.any(valid_mask):
             continue # Skip if no valid segments

        # Apply normalization only where length is non-zero
        perp_dx[valid_mask] /= lengths[valid_mask]
        perp_dy[valid_mask] /= lengths[valid_mask]

        # Ensure component data aligns with the number of segments (n-1 points)
        # If comp_step has n points, take the average or first n-1?
        # Original code used comp_step[:-1], assuming component corresponds to start of segment.
        if len(comp_step) == len(x):
            components = comp_step[:-1] # Align with n-1 segments
        elif len(comp_step) == len(x) - 1:
            components = comp_step # Already aligned
        else:
            print(f"Warning: Mismatch between number of points ({len(x)}) and components ({len(comp_step)}). Skipping vector plotting for this segment.")
            continue

        # Filter components based on valid mask from length calculation
        components = components[valid_mask]
        x_segment_start = x[:-1][valid_mask]
        y_segment_start = y[:-1][valid_mask]
        # Also filter perp_dx, perp_dy which were already partially filtered
        perp_dx = perp_dx[valid_mask]
        perp_dy = perp_dy[valid_mask]

        if len(components) == 0:
             continue # No valid vectors to plot

        max_abs_component = np.max(np.abs(components))
        if max_abs_component == 0:
            # Avoid division by zero if all components are zero
            continue

        # Scale vector lengths based on component magnitude and the new parameter
        # --- MODIFICATION: Used vector_scale_numerator parameter ---
        scale_factor = vector_scale_numerator / max_abs_component
        scaled_magnitudes = components * scale_factor
        u = perp_dx * scaled_magnitudes
        v = perp_dy * scaled_magnitudes

        # Plot vectors (arrows) using quiver
        # --- MODIFICATION: Changed color to 'skyblue' ---
        q = ax.quiver(
            x_segment_start, y_segment_start, # Start points of segments
            u, v,                           # Vector components in projection coordinates
            scale=35,                       # Adjust scale for visual appearance (data units per arrow length unit)
            width=0.0025,                   # Arrow width
            headwidth=0,                    # No arrowhead
            headlength=0,                   # No arrowhead
            headaxislength=0,               # No arrowhead
            color='skyblue',                # Vector color
            transform=proj,                 # Specify coordinates are in the projection system
            zorder=4                        # Draw vectors above trajectory
        )

        # Add quiver key (legend for vectors) only for the first quiver plot
        if first_quiver:
            ax.quiverkey(
                q, X=0.82, Y=0.12, U=0.5, # Position and length of the key arrow (adjust U based on expected scaled values)
                label=f'Scaled Field', # Label - might need adjustment based on vector_scale_numerator meaning
                labelpos='E',           # Label position (East)
                coordinates='axes'      # Position relative to axes
            )
            first_quiver = False # Ensure key is added only once

    # Add title if requested
    if if_title:
        ax.set_title(current_title, fontsize=12, pad=18) # Add padding to avoid overlap with labels

    return fig, ax


def orbits_hemisphere_with_one_vector_projection_custom_colors(
    lons_list: list[np.ndarray],
    lats_list: list[np.ndarray],
    vector_component_list,
    proj_method: str,
    central_longitude: float = 0,
    figsize: tuple[float, float] = (12, 12),
    lon_lat_ext: tuple[float, float, float, float] = None,
    if_cfeature: bool = False,
    if_title: bool = True,
    title: str = None,
    ax: plt.Axes = None,
    step: int = None,
    trajectory_color: str = 'darkgray', # New parameter for trajectory color
    vector_color: str = 'skyblue',      # New parameter for vector color
    vector_scale_numerator: float = 2.0,
):
    """
    Plots satellite orbits and associated vector components perpendicular to the track
    on a North/South Polar Stereographic projection with customizable colors.

    Args:
        lons_list (list[np.ndarray]): List of longitude arrays for each orbit.
        lats_list (list[np.ndarray]): List of latitude arrays for each orbit.
        vector_component_list (list[np.ndarray]): List of scalar component arrays
                                                 for vectors perpendicular to the track.
        proj_method (str): Projection method, either "NorthPolarStereo" or "SouthPolarStereo".
        central_longitude (float, optional): Central longitude for the projection. Defaults to 0.
        figsize (tuple[float, float], optional): Figure size. Defaults to (12, 12).
        lon_lat_ext (tuple[float, float, float, float], optional): Map extent
                                                               (lon_min, lon_max, lat_min, lat_max).
                                                               Defaults based on proj_method.
        if_cfeature (bool, optional): Whether to add Cartopy land/ocean features. Defaults to False.
        if_title (bool, optional): Whether to add a title to the plot. Defaults to True.
        title (str, optional): Custom title string. Defaults based on proj_method.
        ax (plt.Axes, optional): Existing Axes object to plot on. If None, creates a new figure and axes. Defaults to None.
        step (int, optional): Step size for subsampling orbit points and vectors. Defaults to None (no subsampling).
        trajectory_color (str, optional): Color for the orbit trajectory lines. Defaults to 'darkgray'.
        vector_color (str, optional): Color for the perpendicular vectors. Defaults to 'skyblue'.
        vector_scale_numerator (float, optional): Numerator used for scaling vector lengths
                                                 (scale = numerator / max_abs_component). Defaults to 2.0.
    Returns:
        tuple[plt.Figure, plt.Axes]: The figure and axes objects.
    """
    assert proj_method in ["NorthPolarStereo", "SouthPolarStereo"], \
        "proj_method must be either 'NorthPolarStereo' or 'SouthPolarStereo'"

    # Projection configuration
    proj_dict = {
        "NorthPolarStereo": ccrs.NorthPolarStereo,
        "SouthPolarStereo": ccrs.SouthPolarStereo,
    }
    proj = proj_dict[proj_method](central_longitude=central_longitude)
    geodetic = ccrs.PlateCarree()

    # Set default extent, title, and ylocs based on projection
    if proj_method == "NorthPolarStereo":
        default_ext = (-180, 180, 0, 90)
        default_title = "North Hemisphere Map using NorthPolarStereo Projection"
        ylocs = np.arange(0, 91, 15)
    else: # SouthPolarStereo
        default_ext = (-180, 180, -90, 0)
        default_title = "South Hemisphere Map using SouthPolarStereo Projection"
        ylocs = np.arange(-90, 0, 15)

    current_ext = lon_lat_ext if lon_lat_ext is not None else default_ext
    current_title = title if title is not None else default_title

    # Create figure/axes if 'ax' is not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": proj})
    else:
        fig = ax.figure # Use the figure associated with the provided axes

    # Set map extent
    ax.set_extent(list(current_ext), crs=geodetic)

    # Create circular boundary for the polar plot
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

    # Add geographic features if requested
    if if_cfeature:
        ax.add_feature(cfeature.LAND.with_scale("50m"), alpha=0.6, zorder=0, facecolor='lightgray') # Example: subtle land color
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), alpha=0.4, zorder=0, facecolor='aliceblue') # Example: subtle ocean color
        ax.coastlines(resolution="50m", linewidth=0.5, color='gray', zorder=1)

    # Add gridlines
    gl = ax.gridlines(
        crs=geodetic,
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.7,
        linestyle='--',
        xlocs=np.arange(-180, 181, 45),
        ylocs=ylocs,
    )
    gl.top_labels, gl.right_labels, gl.rotate_labels = False, False, False
    # Optional: Use formatters for nicer labels
    # gl.xformatter = LONGITUDE_FORMATTER
    # gl.yformatter = LATITUDE_FORMATTER

    first_quiver = True  # Control quiverkey display (only once)
    for lons, lats, vector_component in zip(lons_list, lats_list, vector_component_list):
        # Apply step for subsampling if specified
        if step is not None and step > 1:
             lons_step = lons[::step]
             lats_step = lats[::step]
             comp_step = vector_component[::step]
        else:
             lons_step = lons
             lats_step = lats
             comp_step = vector_component

        # Transform coordinates to map projection
        try:
            proj_points = proj.transform_points(geodetic, lons_step, lats_step)
            x = proj_points[:, 0]
            y = proj_points[:, 1]
        except Exception as e:
            print(f"Warning: Coordinate transformation failed for an orbit segment: {e}")
            continue # Skip this orbit segment if transformation fails

        # Plot trajectory line
        # --- MODIFICATION: Use trajectory_color parameter ---
        ax.plot(x, y, color=trajectory_color, linestyle='-', lw=1.5, zorder=2, transform=proj)

        # Calculate vector direction (perpendicular to trajectory)
        if len(x) < 2:
            continue
        dx = np.diff(x)
        dy = np.diff(y)
        perp_dx = -dy
        perp_dy = dx

        # Normalize perpendicular direction vectors
        lengths = np.hypot(perp_dx, perp_dy)
        valid_mask = lengths != 0
        if not np.any(valid_mask):
             continue

        perp_dx[valid_mask] /= lengths[valid_mask]
        perp_dy[valid_mask] /= lengths[valid_mask]

        # Align component data with segments
        if len(comp_step) == len(x):
            components = comp_step[:-1]
        elif len(comp_step) == len(x) - 1:
            components = comp_step
        else:
            print(f"Warning: Mismatch between number of points ({len(x)}) and components ({len(comp_step)}). Skipping vector plotting.")
            continue

        # Filter components and coordinates based on valid segments
        components = components[valid_mask]
        x_segment_start = x[:-1][valid_mask]
        y_segment_start = y[:-1][valid_mask]
        perp_dx = perp_dx[valid_mask]
        perp_dy = perp_dy[valid_mask]

        if len(components) == 0:
             continue

        max_abs_component = np.max(np.abs(components))
        if max_abs_component == 0:
            continue

        # Scale vector lengths
        scale_factor = vector_scale_numerator / max_abs_component
        scaled_magnitudes = components * scale_factor
        u = perp_dx * scaled_magnitudes
        v = perp_dy * scaled_magnitudes

        # Plot vectors (arrows) using quiver
        # --- MODIFICATION: Use vector_color parameter ---
        q = ax.quiver(
            x_segment_start, y_segment_start,
            u, v,
            scale=35,          # Adjust scale if needed based on vector_scale_numerator and data range
            width=0.0025,
            headwidth=0,
            headlength=0,
            headaxislength=0,
            color=vector_color, # Use the parameter here
            transform=proj,
            zorder=4
        )

        # Add quiver key (legend for vectors) only for the first quiver plot
        if first_quiver:
            # Adjust U in quiverkey based on expected visual scale if needed
            key_length_example = 0.5 * vector_scale_numerator # Example length in scaled units
            ax.quiverkey(
                q, X=0.9, Y=0.1, U=key_length_example,
                label=f'Scaled Field', # Consider a more specific label if units are known
                labelpos='E',
                coordinates='axes'
            )
            first_quiver = False # Ensure key is added only once

    # Add title if requested
    if if_title:
        ax.set_title(current_title, fontsize=12, pad=18)

    return fig, ax


def orbits_hemisphere_with_uv_vectors_projection(
    lons_list: list[np.ndarray],
    lats_list: list[np.ndarray],
    vector_east_component_list: list[np.ndarray], # Input East component
    vector_north_component_list: list[np.ndarray], # Input North component
    proj_method: str,
    central_longitude: float = 0,
    figsize: tuple[float, float] = (12, 12),
    lon_lat_ext: tuple[float, float, float, float] = None,
    if_cfeature: bool = False,
    if_title: bool = True,
    title: str = None,
    ax: plt.Axes = None,
    step: int = 1, # Default step to 1 (no skipping unless specified)
    trajectory_color: str = 'darkgray',
    vector_color: str = 'skyblue',
    vector_units_label: str = 'units', # Label for quiver key units
    quiver_scale: float = 10.0,       # Scale for quiver (data units per arrow length unit). Adjust based on vector magnitudes.
    quiver_key_magnitude: float = None, # Magnitude reference for the quiver key. If None, uses max magnitude.
    quiver_width: float = 0.003,       # Width of the vector arrows
):
    """
    Plots satellite orbits and associated vectors (defined by East/North components)
    on a North/South Polar Stereographic projection with customizable colors.

    Args:
        lons_list (list[np.ndarray]): List of longitude arrays for each orbit.
        lats_list (list[np.ndarray]): List of latitude arrays for each orbit.
        vector_east_component_list (list[np.ndarray]): List of East component arrays
                                                     for vectors (e.g., U wind, East B-field).
        vector_north_component_list (list[np.ndarray]): List of North component arrays
                                                      for vectors (e.g., V wind, North B-field).
        proj_method (str): Projection method, "NorthPolarStereo" or "SouthPolarStereo".
        central_longitude (float, optional): Central longitude for the projection. Defaults to 0.
        figsize (tuple[float, float], optional): Figure size. Defaults to (12, 12).
        lon_lat_ext (tuple[float, float, float, float], optional): Map extent. Defaults based on proj_method.
        if_cfeature (bool, optional): Add Cartopy land/ocean features. Defaults to False.
        if_title (bool, optional): Add a title. Defaults to True.
        title (str, optional): Custom title string. Defaults based on proj_method.
        ax (plt.Axes, optional): Existing Axes object. If None, creates new figure/axes. Defaults to None.
        step (int, optional): Step size for subsampling orbit points/vectors. Defaults to 1.
        trajectory_color (str, optional): Color for orbit lines. Defaults to 'darkgray'.
        vector_color (str, optional): Color for vectors. Defaults to 'skyblue'.
        vector_units_label (str, optional): Units label for quiver key. Defaults to 'units'.
        quiver_scale (float, optional): Quiver scale factor. Larger values make arrows smaller.
                                        Adjust based on expected vector magnitudes. Defaults to 500.0.
        quiver_key_magnitude (float, optional): Magnitude for the reference vector in the key.
                                                If None, uses the maximum magnitude found. Defaults to None.
        quiver_width (float, optional): Width of the quiver arrows in figure fraction units. Defaults to 0.003.

    Returns:
        tuple[plt.Figure, plt.Axes]: The figure and axes objects.
    """
    assert proj_method in ["NorthPolarStereo", "SouthPolarStereo"], \
        "proj_method must be either 'NorthPolarStereo' or 'SouthPolarStereo'"
    assert len(lons_list) == len(lats_list) == len(vector_east_component_list) == len(vector_north_component_list), \
        "Input lists (lons, lats, east_comp, north_comp) must have the same length."

    # Projection configuration
    proj_dict = {
        "NorthPolarStereo": ccrs.NorthPolarStereo,
        "SouthPolarStereo": ccrs.SouthPolarStereo,
    }
    proj = proj_dict[proj_method](central_longitude=central_longitude)
    geodetic = ccrs.PlateCarree() # Input data CRS is Plate Carree (lon/lat)

    # Set default extent, title, and ylocs based on projection
    if proj_method == "NorthPolarStereo":
        default_ext = (-180, 180, 0, 90)
        default_title = "North Hemisphere Map using NorthPolarStereo Projection"
        ylocs = np.arange(0, 91, 15)
        # Limit latitude extent slightly from pole for better visualization if needed
        if lon_lat_ext is None: lon_lat_ext = (-180, 180, 10, 90) # Example limit
    else: # SouthPolarStereo
        default_ext = (-180, 180, -90, 0)
        default_title = "South Hemisphere Map using SouthPolarStereo Projection"
        ylocs = np.arange(-90, 0, 15)
        if lon_lat_ext is None: lon_lat_ext = (-180, 180, -90, -10) # Example limit

    current_ext = lon_lat_ext if lon_lat_ext is not None else default_ext
    current_title = title if title is not None else default_title

    # Create figure/axes if 'ax' is not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": proj})
    else:
        fig = ax.figure # Use the figure associated with the provided axes

    # Set map extent
    ax.set_extent(list(current_ext), crs=geodetic)

    # Create circular boundary for the polar plot
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

    # Add geographic features if requested
    if if_cfeature:
        ax.add_feature(cfeature.LAND.with_scale("50m"), alpha=0.6, zorder=0, facecolor='lightgray')
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), alpha=0.4, zorder=0, facecolor='aliceblue')
        ax.coastlines(resolution="50m", linewidth=0.5, color='gray', zorder=1)

    # Add gridlines
    gl = ax.gridlines(
        crs=geodetic, draw_labels=True, linewidth=0.5, color="gray",
        alpha=0.7, linestyle='--', xlocs=np.arange(-180, 181, 45), ylocs=ylocs,
    )
    gl.top_labels, gl.right_labels, gl.rotate_labels = False, False, False
    # Optional: Formatters for nicer labels
    # gl.xformatter = LONGITUDE_FORMATTER
    # gl.yformatter = LATITUDE_FORMATTER

    first_quiver = True  # Control quiverkey display (only once)
    max_magnitude_overall = 0.0 # Track max magnitude for potential key usage

    for i, (lons, lats, u_east, v_north) in enumerate(zip(
        lons_list, lats_list, vector_east_component_list, vector_north_component_list
    )):
        # Check length consistency for this specific orbit
        if not (len(lons) == len(lats) == len(u_east) == len(v_north)):
            print(f"Warning: Skipping orbit {i} due to inconsistent lengths of lon/lat/vector arrays.")
            continue
        if len(lons) < 1:
            continue # Skip empty orbits

        # Apply step for subsampling if specified
        if step is not None and step > 1:
             lons_step = lons[::step]
             lats_step = lats[::step]
             u_east_step = u_east[::step]
             v_north_step = v_north[::step]
        else:
             lons_step = lons
             lats_step = lats
             u_east_step = u_east
             v_north_step = v_north

        if len(lons_step) < 1:
            continue # Skip if subsampling resulted in empty array

        # Transform coordinates to map projection
        try:
            # transform_points needs shape (N, 2) or separate x, y
            proj_points = proj.transform_points(geodetic, lons_step, lats_step)
            x_proj = proj_points[:, 0]
            y_proj = proj_points[:, 1]
        except Exception as e:
            print(f"Warning: Coordinate transformation failed for orbit {i}: {e}")
            continue

        # --- Vector Transformation ---
        # Transform vectors from Geographic (East, North) to Projection coordinates
        try:
            # Ensure inputs to transform_vectors are valid
            valid_vector_mask = np.isfinite(lons_step) & np.isfinite(lats_step) & \
                                np.isfinite(u_east_step) & np.isfinite(v_north_step)

            if not np.any(valid_vector_mask):
                print(f"Warning: No valid vector data for orbit {i} after filtering.")
                continue

            # Apply mask before transformation
            lons_valid = lons_step[valid_vector_mask]
            lats_valid = lats_step[valid_vector_mask]
            u_east_valid = u_east_step[valid_vector_mask]
            v_north_valid = v_north_step[valid_vector_mask]
            x_proj_valid = x_proj[valid_vector_mask]
            y_proj_valid = y_proj[valid_vector_mask]


            u_proj, v_proj = proj.transform_vectors(
                geodetic, lons_valid, lats_valid, u_east_valid, v_north_valid
            )
        except Exception as e:
            print(f"Warning: Vector transformation failed for orbit {i}: {e}")
            # Still plot trajectory if points are valid
            if len(x_proj) > 0:
                 ax.plot(x_proj, y_proj, color=trajectory_color, linestyle='-', lw=1.5, zorder=2, transform=proj)
            continue # Skip vector plotting for this orbit

        # Update overall max magnitude (using original components for physical meaning)
        current_mags = np.hypot(u_east_valid, v_north_valid)
        if len(current_mags) > 0:
             max_magnitude_overall = max(max_magnitude_overall, np.max(current_mags))

        # Plot trajectory line (use original potentially non-masked points for continuity)
        if len(x_proj) > 0:
             ax.plot(x_proj, y_proj, color=trajectory_color, linestyle='-', lw=1.5, zorder=2, transform=proj)


        # Plot vectors using quiver
        # Note: x_proj_valid, y_proj_valid, u_proj, v_proj are all in the projection's coordinate system.
        # We do NOT use transform=proj here because the coordinates and vectors are already transformed.
        q = ax.quiver(
            x_proj_valid, y_proj_valid, # Vector origins in projection coordinates
            u_proj, v_proj,             # Vector components in projection coordinates
            color=vector_color,
            scale=quiver_scale,         # Controls arrow size (data units per arrow length unit)
            # scale_units='xy',           # Scale applies equally to x and y  # result in no length of vector, just some dots. So not set,use None.
            angles='xy',                # Angles are relative to plot axes
            width=quiver_width,         # Arrow width
            headwidth=0,
            headlength=0,
            headaxislength=0,
            zorder=4                    # Draw vectors above trajectory
            # headwidth=3, headlength=5 # Optional: Add arrowheads if desired
        )

        # Add quiver key (legend for vectors) only for the first valid quiver plot
        if first_quiver and len(x_proj_valid) > 0:
            # Determine the magnitude for the key
            if quiver_key_magnitude is None:
                # Use a representative value like max overall or a percentile if max is too extreme
                key_mag = max_magnitude_overall if max_magnitude_overall > 0 else 1.0
            else:
                key_mag = quiver_key_magnitude

            ax.quiverkey(
                q,
                X=0.85, Y=0.10,           # Position of the key (adjust as needed)
                U=key_mag,                # The magnitude the key represents (in original vector units)
                label=f'{key_mag:.1f} {vector_units_label}', # Label for the key
                labelpos='E',             # Label position (East)
                coordinates='axes',       # Position relative to axes
                fontproperties={'size': 9} # Adjust font size if needed
            )
            first_quiver = False # Ensure key is added only once

    # Add title if requested
    if if_title:
        ax.set_title(current_title, fontsize=12, pad=18)

    return fig, ax


# Optional: For prettier tick labels
# from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# --- Internal Helper Function for Plotting a Single Hemisphere ---
def _plot_hemisphere_internal(
    ax: plt.Axes,
    proj: ccrs.Projection,
    proj_method: str,
    lons_list: list[np.ndarray],
    lats_list: list[np.ndarray],
    vector_east_component_list: list[np.ndarray],
    vector_north_component_list: list[np.ndarray],
    central_longitude: float,
    lon_lat_ext: tuple[float, float, float, float],
    if_cfeature: bool,
    step: int,
    trajectory_color: str,
    vector_color: str,
    vector_units_label: str,
    quiver_scale: float,
    # quiver_key_magnitude is NOW directly used if add_quiver_key is True
    quiver_key_magnitude: float,
    quiver_width: float,
    add_quiver_key: bool = True,
    quiver_key_X: float = 0.85,
    quiver_key_Y: float = 0.10,
):
    """Internal helper to plot data on a single polar axes."""

    geodetic = ccrs.PlateCarree()

    # --- (Extent, Boundary, Features, Gridlines - same as before) ---
    if proj_method == "NorthPolarStereo":
        default_ext = (-180, 180, 0, 90)
        ylocs = np.arange(0, 91, 15)
        if lon_lat_ext is None: lon_lat_ext = (-180, 180, 10, 90)
    else: # SouthPolarStereo
        default_ext = (-180, 180, -90, 0)
        ylocs = np.arange(-90, 0, 15)
        if lon_lat_ext is None: lon_lat_ext = (-180, 180, -90, -10)
    current_ext = lon_lat_ext
    ax.set_extent(list(current_ext), crs=geodetic)
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)
    if if_cfeature:
        ax.add_feature(cfeature.LAND.with_scale("50m"), alpha=0.6, zorder=0, facecolor='lightgray')
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), alpha=0.4, zorder=0, facecolor='aliceblue')
        ax.coastlines(resolution="50m", linewidth=0.5, color='gray', zorder=1)
    gl = ax.gridlines(
        crs=geodetic, draw_labels=True, linewidth=0.5, color="gray",
        alpha=0.7, linestyle='--', xlocs=np.arange(-180, 181, 45), ylocs=ylocs,
    )
    gl.top_labels, gl.right_labels, gl.rotate_labels = False, False, False
    # --- (End Setup) ---

    q_return = None # Store last quiver object

    for i, (lons, lats, u_east, v_north) in enumerate(zip(
        lons_list, lats_list, vector_east_component_list, vector_north_component_list
    )):
        # --- (Data validation, subsampling, transformations - same as before) ---
        if not (len(lons) == len(lats) == len(u_east) == len(v_north)):
            print(f"Warning: Skipping orbit {i} in {proj_method} due to inconsistent lengths.")
            continue
        if len(lons) < 1: continue
        if step is not None and step > 1:
             lons_step, lats_step = lons[::step], lats[::step]
             u_east_step, v_north_step = u_east[::step], v_north[::step]
        else:
             lons_step, lats_step = lons, lats
             u_east_step, v_north_step = u_east, v_north
        if len(lons_step) < 1: continue
        try:
            proj_points = proj.transform_points(geodetic, lons_step, lats_step)
            x_proj, y_proj = proj_points[:, 0], proj_points[:, 1]
        except Exception as e:
            print(f"Warning: Coord transform failed for orbit {i} in {proj_method}: {e}")
            continue
        try:
            valid_mask = np.isfinite(lons_step) & np.isfinite(lats_step) & \
                         np.isfinite(u_east_step) & np.isfinite(v_north_step)
            if not np.any(valid_mask): continue
            lons_valid, lats_valid = lons_step[valid_mask], lats_step[valid_mask]
            u_east_valid, v_north_valid = u_east_step[valid_mask], v_north_step[valid_mask]
            x_proj_valid, y_proj_valid = x_proj[valid_mask], y_proj[valid_mask]
            u_proj, v_proj = proj.transform_vectors(
                geodetic, lons_valid, lats_valid, u_east_valid, v_north_valid
            )
        except Exception as e:
            print(f"Warning: Vector transform failed for orbit {i} in {proj_method}: {e}")
            if len(x_proj) > 0:
                 ax.plot(x_proj, y_proj, color=trajectory_color, linestyle='-', lw=1.5, zorder=2, transform=proj)
            continue
        # --- (End Transformations) ---

        # --- (Plotting trajectory and vectors - same as before) ---
        if len(x_proj) > 0:
            ax.plot(x_proj, y_proj, color=trajectory_color, linestyle='-', lw=1.5, zorder=2, transform=proj)
        q = ax.quiver(
            x_proj_valid, y_proj_valid, u_proj, v_proj,
            color=vector_color, scale=quiver_scale, angles='xy', width=quiver_width,
            headwidth=0, headlength=0, headaxislength=0, zorder=4
        )
        q_return = q
        # --- (End Plotting) ---

    # --- Quiver Key Logic (Simplified) ---
    # Add key ONLY if requested, vectors were plotted, and key magnitude is provided
    if add_quiver_key and q_return is not None and quiver_key_magnitude is not None:
        ax.quiverkey(
            q_return,
            X=quiver_key_X,
            Y=quiver_key_Y,
            U=quiver_key_magnitude, # Directly use the provided value
            label=f'{quiver_key_magnitude:.1f} {vector_units_label}',
            labelpos='E', coordinates='axes', fontproperties={'size': 9}
        )
    elif add_quiver_key and q_return is not None and quiver_key_magnitude is None:
         print("Warning: Quiver key requested but 'quiver_key_magnitude' not provided. Key not added.")
    # --- End Quiver Key Logic ---


# --- Main Function to Create Dual Hemisphere Plot ---
def plot_dual_hemisphere_orbits(
    # North Data
    lons_nor_list: list[np.ndarray],
    lats_nor_list: list[np.ndarray],
    vector_east_component_nor_list: list[np.ndarray],
    vector_north_component_nor_list: list[np.ndarray],
    # South Data
    lons_sou_list: list[np.ndarray],
    lats_sou_list: list[np.ndarray],
    vector_east_component_sou_list: list[np.ndarray],
    vector_north_component_sou_list: list[np.ndarray],
    # --- REQUIRED Quiver Key Magnitude ---
    quiver_key_magnitude: float, # Make this a required argument
    # Common Parameters
    central_longitude: float = 0,
    figsize: tuple[float, float] = (20, 10),
    lon_lat_ext_nor: tuple[float, float, float, float] = None,
    lon_lat_ext_sou: tuple[float, float, float, float] = None,
    if_cfeature: bool = False,
    main_title: str = "Hemispheric Orbit Data",
    step: int = 1,
    trajectory_color: str = 'darkgray',
    vector_color: str = 'skyblue',
    vector_units_label: str = 'units',
    quiver_scale: float = 10.0,
    # quiver_key_magnitude is now required
    quiver_key_X: float = 0.85,
    quiver_key_Y: float = 0.10,
    quiver_width: float = 0.003,
):
    """
    Plots satellite orbits and vectors for North and South hemispheres side-by-side.
    The magnitude represented by the quiver key MUST be provided via 'quiver_key_magnitude'.

    Args:
        ... (lons/lats/vector lists for North and South) ...
        quiver_key_magnitude (float): The magnitude the reference vector in the key represents. **Required**.
        central_longitude (float, optional): Central longitude. Defaults to 0.
        figsize (tuple[float, float], optional): Overall figure size. Defaults to (20, 10).
        lon_lat_ext_nor (tuple, optional): Extent override for North plot. Defaults internally.
        lon_lat_ext_sou (tuple, optional): Extent override for South plot. Defaults internally.
        if_cfeature (bool, optional): Add Cartopy features. Defaults to False.
        main_title (str, optional): Title for the figure. Defaults to "Hemispheric Orbit Data".
        step (int, optional): Subsampling step. Defaults to 1.
        trajectory_color (str, optional): Trajectory color. Defaults to 'darkgray'.
        vector_color (str, optional): Vector color. Defaults to 'skyblue'.
        vector_units_label (str, optional): Units label for key. Defaults to 'units'.
        quiver_scale (float, optional): Quiver scale factor. Adjust based on data! Defaults to 10.0.
        quiver_key_X (float, optional): X coordinate (axes fraction) for quiver key. Defaults to 0.85.
        quiver_key_Y (float, optional): Y coordinate (axes fraction) for quiver key. Defaults to 0.10.
        quiver_width (float, optional): Vector width. Defaults to 0.003.

    Returns:
        tuple[plt.Figure, np.ndarray[plt.Axes]]: The figure and array containing the two axes objects.
    """

    # --- (Projection and Figure/Axes Setup - same as before) ---
    proj_nor = ccrs.NorthPolarStereo(central_longitude=central_longitude)
    proj_sou = ccrs.SouthPolarStereo(central_longitude=central_longitude)
    fig = plt.figure(figsize=figsize)
    ax_nor = fig.add_subplot(1, 2, 1, projection=proj_nor)
    ax_sou = fig.add_subplot(1, 2, 2, projection=proj_sou)
    axes = np.array([ax_nor, ax_sou])
    # --- (End Setup) ---


    # Plot North Hemisphere (NO key)
    # Pass quiver_key_magnitude=None (or any value, doesn't matter as add_quiver_key=False)
    _plot_hemisphere_internal(
        ax=ax_nor, proj=proj_nor, proj_method="NorthPolarStereo",
        lons_list=lons_nor_list, lats_list=lats_nor_list,
        vector_east_component_list=vector_east_component_nor_list,
        vector_north_component_list=vector_north_component_nor_list,
        central_longitude=central_longitude, lon_lat_ext=lon_lat_ext_nor,
        if_cfeature=if_cfeature, step=step,
        trajectory_color=trajectory_color, vector_color=vector_color,
        vector_units_label=vector_units_label, quiver_scale=quiver_scale,
        quiver_key_magnitude=None, # Value ignored since add_quiver_key=False
        quiver_width=quiver_width,
        add_quiver_key=False,
        # X/Y don't matter here
    )
    ax_nor.set_title("Northern Hemisphere", fontsize=11, pad=15)


    # Plot South Hemisphere (WITH key, using the REQUIRED magnitude)
    _plot_hemisphere_internal(
        ax=ax_sou, proj=proj_sou, proj_method="SouthPolarStereo",
        lons_list=lons_sou_list, lats_list=lats_sou_list,
        vector_east_component_list=vector_east_component_sou_list,
        vector_north_component_list=vector_north_component_sou_list,
        central_longitude=central_longitude, lon_lat_ext=lon_lat_ext_sou,
        if_cfeature=if_cfeature, step=step,
        trajectory_color=trajectory_color, vector_color=vector_color,
        vector_units_label=vector_units_label, quiver_scale=quiver_scale,
        quiver_key_magnitude=quiver_key_magnitude, # Pass the REQUIRED value
        quiver_width=quiver_width,
        add_quiver_key=True,
        quiver_key_X=quiver_key_X,
        quiver_key_Y=quiver_key_Y,
    )
    ax_sou.set_title("Southern Hemisphere", fontsize=11, pad=15)

    # --- (Suptitle and Layout - same as before) ---
    if main_title:
        fig.suptitle(main_title, fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    # --- (End Layout) ---

    return fig, axes

# --- Example Usage ---
# fig, axes = plot_dual_hemisphere_orbits(
#     # ... (provide your data lists) ...
#     # --- Provide the key magnitude EXPLICITLY ---
#     quiver_key_magnitude=100.0,
#     # --- Other parameters ---
#     if_cfeature=True,
#     step=5,
#     main_title="Example Dual Hemisphere Orbits (User-defined Key Mag)",
#     vector_units_label="nT",
#     quiver_scale=500.0, # REMEMBER TO ADJUST THIS!
#     quiver_key_X=0.85,
#     quiver_key_Y=0.10,
# )
# plt.show()