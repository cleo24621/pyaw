import numpy as np
from cartopy import crs as ccrs
from matplotlib import pyplot as plt, path as mpath
from matplotlib.axes import Axes
from nptyping import NDArray


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


def orbit_hemisphere_with_vector_projection(lons,lats,vector_east,vector_north,proj_method,central_longitude=0,zorder:float=2,):
    fig,ax = orbit_hemisphere_projection(lons=lons,lats=lats,proj_method=proj_method)
    proj_dict = {
        "NorthPolarStereo": ccrs.NorthPolarStereo,
        "SouthPolarStereo": ccrs.SouthPolarStereo,
    }
    proj = proj_dict[proj_method](central_longitude=central_longitude)
    geodetic = ccrs.PlateCarree()

    u_proj, v_proj = proj.transform_vectors(
        geodetic,
        lons, lats,
        vector_east, vector_north
    )
    q = ax.quiver(
        lons, lats,
        u_proj, v_proj,
        scale=35,  # 增大scale值以适应更大范围
        width=0.0025,  # 更细的箭头杆
        headwidth=0,
        headlength=0,
        headaxislength=0,  # 箭头轴长度（可选）(default=4.5，设置不同值的影响查看示例)（此绘制最好设置为0）
        color='crimson',
        transform=geodetic,
        zorder=zorder
    )
    ax.quiverkey(
        q, X=0.82, Y=0.12, U=0.3,
        label=f'Normalized resultant vector of the eastward and northward components', labelpos='E',  # 标签在箭头右侧
        coordinates='axes'
    )
    return fig,ax

