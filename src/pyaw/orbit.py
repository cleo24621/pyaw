import cartopy.feature as cfeature
import numpy as np
from cartopy import crs as ccrs
from matplotlib import pyplot as plt, path as mpath
from numpy.typing import NDArray


def get_nor_sou_split_indices_swarm_dmsp(latitudes: NDArray):
    """Get the indices that split the 1D array of latitudes into northern and southern.

    Args:
        latitudes: pattern is [south,north], or [south,north,south(few)]

    Notes:
        Pay attention to the case that some latitudes are exactly 0.
    """
    neg_indices = np.where(latitudes < 0)[0]
    if not neg_indices.size:
        return (0, len(latitudes)), (len(latitudes), len(latitudes))

    start_south = neg_indices[0]
    pos_indices = np.where(latitudes[start_south:] >= 0)[0]
    end_south = start_south + pos_indices[0] if pos_indices.size else len(latitudes)

    return (0, start_south), (start_south, end_south)


class GetZh1NorSouSplitIndices:
    def __init__(self, file_name):
        """

        Args:
            file_name: support the filename of EFD 2A data product.
        """
        self.file_name = file_name
        self.orbit_number, self.indicator, self.start_time, self.end_time = (
            self._get_orbit_number_indicator_st_et()
        )

    def _get_orbit_number_indicator_st_et(self):
        """

        Notes:
            1 for ascending (south to north).
            0 for descending (north to south).
            因为不同2级产品的命名格式是固定的，所以当前方法适用于所有2a级产品的文件名（参考文档）。
        """
        parts = self.file_name.split("_")
        part = parts[6]
        assert part[-1] in ["0", "1"]
        start_time = parts[7] + "_" + parts[8]
        end_time = parts[9] + "_" + parts[10]

        return parts[6][:-1], parts[6][-1], start_time, end_time

    def get_nor_sou_split_indices(self, latitudes: NDArray):
        """Refer to 'get_nor_sou_split_indices_swarm_dmsp()'."""
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


class HemisphereProjection:
    """A class to handle hemisphere projections using Cartopy."""

    def __init__(
        self,
        xlocs: NDArray = np.arange(-180, 181, 45),
        central_longitude: float = 0,
        figsize: tuple[float, float] = (12, 12),
        if_cfeature: bool = True,
    ):
        """

        Args:
        """
        self.fig = None
        self.ax = None
        self.proj = None
        self.geodetic = None
        self.xlocs = xlocs
        self.central_longitude = central_longitude
        self.figsize = figsize
        self.if_cfeature = if_cfeature
        self.config = {
            "north": {
                "extent": [-180, 180, 0, 90],
                "ylocs": np.arange(0, 91, 15),
                "title": "North Hemisphere Map using NorthPolarStereo Projection",
            },
            "south": {
                "extent": [-180, 180, -90, 0],
                "ylocs": np.arange(-90, 0, 15),
                "title": "South Hemisphere Map using SouthPolarStereo Projection",
            },
        }

    def base_projection(self, _type: str = "north"):
        """Create a base projection for hemisphere maps."""
        if _type == "north":
            self.proj = ccrs.NorthPolarStereo(central_longitude=self.central_longitude)
            extent = self.config["north"]["extent"]
            ylocs = self.config["north"]["ylocs"]
            title = self.config["north"]["title"]

        elif _type == "south":
            self.proj = ccrs.SouthPolarStereo(central_longitude=self.central_longitude)
            extent = self.config["south"]["extent"]
            ylocs = self.config["south"]["ylocs"]
            title = self.config["south"]["title"]
        else:
            raise ValueError("Type must be 'north' or 'south'.")
        self.geodetic = ccrs.PlateCarree()
        self.fig, self.ax = plt.subplots(
            figsize=self.figsize, subplot_kw={"projection": self.proj}
        )
        self.ax = self._set_border(self.ax, extent, self.geodetic)
        if self.if_cfeature:
            self.ax = self._add_cfeature(self.ax)
        self.ax = self._add_gridlines(self.ax, self.xlocs, ylocs)
        self.ax.set_title(title, fontsize=12, pad=18)

        return self.fig, self.ax

    @staticmethod
    def _set_border(ax, extent, geodetic):
        # Set border
        ax.set_extent(extent, crs=geodetic)
        # Create circle border
        theta = np.linspace(0, 2 * np.pi, 100)
        circle = mpath.Path(
            np.column_stack([np.sin(theta), np.cos(theta)]) * 0.45 + 0.5
        )
        ax.set_boundary(circle, transform=ax.transAxes)

        return ax

    @staticmethod
    def _add_cfeature(ax):
        ax.add_feature(cfeature.LAND.with_scale("50m"), alpha=0.6)
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), alpha=0.4)
        ax.coastlines(resolution="50m", linewidth=0.5)

        return ax

    @staticmethod
    def _add_gridlines(ax, xlocs: NDArray, ylocs: NDArray):
        """Add gridlines to the axes."""
        # Add gridlines
        gl = ax.gridlines(
            draw_labels=True,
            linewidth=0.5,
            color="gray",
            xlocs=xlocs,
            ylocs=ylocs,
            xpadding=15,
            ypadding=15,
        )
        gl.top_labels, gl.right_labels, gl.rotate_labels = False, False, False

        return ax

    def add_orbit(self, lons: NDArray, lats: NDArray, satellite: str = None):
        self.ax.plot(
            *self.proj.transform_points(self.geodetic, lons, lats)[:, :2].T,
            "b-",
            lw=1.5,
            label="Satellite orbit" if satellite is None else f"{satellite} orbit",
        )
        self.ax.legend(loc="upper right")

        return None

    def add_vector(
        self,
        lons: NDArray,
        lats: NDArray,
        vector_east: NDArray,
        vector_north: NDArray,
        scale: float = 35,
        width: float = 0.0025,
        headwidth: float = 0,
        headlength: float = 0,
        color: str = "crimson",
        zorder: int = 2,
    ):
        u_proj, v_proj = self.proj.transform_vectors(
            self.geodetic, lons, lats, vector_east, vector_north
        )
        q = self.ax.quiver(
            lons,
            lats,
            u_proj,
            v_proj,
            scale=scale,
            width=width,
            headwidth=headwidth,
            headlength=headlength,
            color=color,
            transform=self.geodetic,
            zorder=zorder,
        )
        self.ax.quiverkey(
            q,
            X=0.82,
            Y=0.12,
            U=0.3,
            label=f"Normalized resultant vector of the eastward and northward components",
            labelpos="E",  # 标签在箭头右侧
            coordinates="axes",
        )
