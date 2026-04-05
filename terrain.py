from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any

import geopandas as gpd
import mercantile
import numpy as np
import requests
import trimesh
from PIL import Image
from pyproj import Transformer
from shapely.affinity import translate
from shapely.geometry import Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.prepared import prep

from utils import ensure_directory, geometry_sample_coordinates, repair_mesh, safe_exit

try:
    from scipy.ndimage import gaussian_filter
except Exception:  # pragma: no cover - optional dependency
    gaussian_filter = None

try:
    from scipy.spatial import Delaunay
except Exception:  # pragma: no cover - optional dependency
    Delaunay = None


TERRARIUM_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
MAX_TERRAIN_TRIANGULATION_POINTS = 30000
TERRAIN_TILE_CACHE_DIR = ensure_directory("cache/terrain_tiles")


@dataclass(slots=True)
class TerrainContext:
    dem: np.ndarray
    bounds_latlon: tuple[float, float, float, float]
    utm_crs: Any
    boundary_polygon_shifted: BaseGeometry
    x_origin: float
    y_origin: float
    x_grid: np.ndarray
    y_grid: np.ndarray
    x_axis: np.ndarray
    y_axis: np.ndarray
    z_grid: np.ndarray
    scale_factor: float


def terrarium_to_height(rgb: np.ndarray) -> np.ndarray:
    red = rgb[:, :, 0].astype(np.float32)
    green = rgb[:, :, 1].astype(np.float32)
    blue = rgb[:, :, 2].astype(np.float32)
    return (red * 256.0 + green + blue / 256.0) - 32768.0


def fetch_tile(z: int, x: int, y: int) -> np.ndarray:
    url = TERRARIUM_URL.format(z=z, x=x, y=y)
    cache_path = TERRAIN_TILE_CACHE_DIR / str(z) / str(x) / f"{y}.png"
    if cache_path.exists() and cache_path.stat().st_size > 0:
        image = Image.open(cache_path).convert("RGB")
        return np.array(image)

    last_error: Exception | None = None
    for timeout_seconds in (45, 90):
        try:
            response = requests.get(url, timeout=timeout_seconds)
            response.raise_for_status()
            ensure_directory(cache_path.parent)
            cache_path.write_bytes(response.content)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            return np.array(image)
        except Exception as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Terrain tile alinamadi: z{z}/{x}/{y}")


def build_dem_from_tiles(
    west: float,
    south: float,
    east: float,
    north: float,
    zoom: int = 12,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    tiles = list(mercantile.tiles(west, south, east, north, [zoom]))
    if not tiles:
        safe_exit("Terrain tile bulunamadi")

    min_x = min(tile.x for tile in tiles)
    max_x = max(tile.x for tile in tiles)
    min_y = min(tile.y for tile in tiles)
    max_y = max(tile.y for tile in tiles)

    tile_count_x = max_x - min_x + 1
    tile_count_y = max_y - min_y + 1
    print(f"Terrain tile sayisi: {len(tiles)} ({tile_count_x} x {tile_count_y})")

    mosaic = np.zeros((tile_count_y * 256, tile_count_x * 256), dtype=np.float32)
    failed_tiles: list[str] = []
    fetched_tiles = 0

    for tile in tiles:
        try:
            heights = terrarium_to_height(fetch_tile(tile.z, tile.x, tile.y))
            row = tile.y - min_y
            col = tile.x - min_x
            y0 = row * 256
            x0 = col * 256
            mosaic[y0 : y0 + 256, x0 : x0 + 256] = heights
            fetched_tiles += 1
        except Exception as exc:
            print(f"Tile alinamadi: z{tile.z}/{tile.x}/{tile.y} -> {exc}")
            failed_tiles.append(f"z{tile.z}/{tile.x}/{tile.y}")

    if fetched_tiles == 0:
        safe_exit("Terrain verisi alinamadi; duz arazi uretmeyi durdurdum.")

    if failed_tiles:
        safe_exit(
            "Terrain verisi eksik oldugu icin islem durduruldu. Eksik tile'lar: "
            + ", ".join(failed_tiles[:6])
        )

    top_left = mercantile.bounds(min_x, min_y, zoom)
    bottom_right = mercantile.bounds(max_x, max_y, zoom)
    return mosaic, (top_left.west, bottom_right.south, bottom_right.east, top_left.north)


def crop_dem_to_bbox(
    dem: np.ndarray,
    dem_bounds: tuple[float, float, float, float],
    target_bounds: tuple[float, float, float, float],
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    dem_west, dem_south, dem_east, dem_north = dem_bounds
    west, south, east, north = target_bounds

    rows, cols = dem.shape
    x_res = (dem_east - dem_west) / cols
    y_res = (dem_north - dem_south) / rows

    col0 = max(0, int((west - dem_west) / x_res))
    col1 = min(cols, int((east - dem_west) / x_res) + 1)
    row0 = max(0, int((dem_north - north) / y_res))
    row1 = min(rows, int((dem_north - south) / y_res) + 1)

    cropped = dem[row0:row1, col0:col1]
    cropped_west = dem_west + col0 * x_res
    cropped_east = dem_west + col1 * x_res
    cropped_north = dem_north - row0 * y_res
    cropped_south = dem_north - row1 * y_res
    return cropped, (cropped_west, cropped_south, cropped_east, cropped_north)


def downsample_dem(dem: np.ndarray, max_size: int = 240) -> np.ndarray:
    rows, cols = dem.shape
    step = max(1, int(np.ceil(max(rows, cols) / max_size)))
    return dem[::step, ::step]


def smooth_dem(dem: np.ndarray, sigma: float = 0.8) -> np.ndarray:
    if sigma <= 0 or gaussian_filter is None:
        return dem
    return gaussian_filter(dem, sigma=sigma)


def build_terrain_context(
    boundary_polygon_latlon: BaseGeometry,
    utm_crs: Any,
    scale_factor: float,
    zoom: int = 12,
    max_size: int = 240,
    z_scale: float = 1.6,
    smoothing_sigma: float = 0.8,
) -> TerrainContext:
    west, south, east, north = boundary_polygon_latlon.bounds
    dem, dem_bounds = build_dem_from_tiles(west, south, east, north, zoom=zoom)
    dem, cropped_bounds = crop_dem_to_bbox(dem, dem_bounds, (west, south, east, north))
    dem = downsample_dem(dem, max_size=max_size)
    dem = smooth_dem(dem, sigma=smoothing_sigma)

    rows, cols = dem.shape
    xs = np.linspace(cropped_bounds[0], cropped_bounds[2], cols)
    ys = np.linspace(cropped_bounds[3], cropped_bounds[1], rows)
    lon_grid, lat_grid = np.meshgrid(xs, ys)

    transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    x_grid, y_grid = transformer.transform(lon_grid, lat_grid)
    x_grid = np.asarray(x_grid, dtype=np.float64)
    y_grid = np.asarray(y_grid, dtype=np.float64)

    x_origin = float(np.min(x_grid))
    y_origin = float(np.min(y_grid))
    x_grid = x_grid - x_origin
    y_grid = y_grid - y_origin

    boundary_projected = gpd.GeoSeries([boundary_polygon_latlon], crs="EPSG:4326").to_crs(utm_crs).iloc[0]
    boundary_polygon_shifted = translate(boundary_projected, xoff=-x_origin, yoff=-y_origin)
    z_grid = np.nan_to_num(dem, nan=float(np.nanmean(dem))) * z_scale

    return TerrainContext(
        dem=dem,
        bounds_latlon=cropped_bounds,
        utm_crs=utm_crs,
        boundary_polygon_shifted=boundary_polygon_shifted,
        x_origin=x_origin,
        y_origin=y_origin,
        x_grid=x_grid,
        y_grid=y_grid,
        x_axis=x_grid[0, :],
        y_axis=y_grid[:, 0],
        z_grid=z_grid,
        scale_factor=scale_factor,
    )


def _nearest_axis_index(axis: np.ndarray, value: float) -> int:
    if axis.size == 1:
        return 0

    increasing = axis[0] <= axis[-1]
    if increasing:
        index = int(np.searchsorted(axis, value))
        if index <= 0:
            return 0
        if index >= axis.size:
            return axis.size - 1
        before = axis[index - 1]
        after = axis[index]
        return index - 1 if abs(value - before) <= abs(after - value) else index

    reversed_axis = axis[::-1]
    index = int(np.searchsorted(reversed_axis, value))
    if index <= 0:
        return axis.size - 1
    if index >= axis.size:
        return 0
    before = reversed_axis[index - 1]
    after = reversed_axis[index]
    resolved = index - 1 if abs(value - before) <= abs(after - value) else index
    return axis.size - 1 - resolved


def sample_elevation(context: TerrainContext, x: float, y: float) -> float:
    if context.x_axis.size == 1 or context.y_axis.size == 1:
        col = _nearest_axis_index(context.x_axis, x)
        row = _nearest_axis_index(context.y_axis, y)
        return float(context.z_grid[row, col])

    if context.x_axis[0] <= context.x_axis[-1]:
        col_f = float(
            np.interp(
                x,
                context.x_axis,
                np.arange(context.x_axis.size, dtype=np.float64),
                left=0.0,
                right=float(context.x_axis.size - 1),
            )
        )
    else:
        col_f = float(
            np.interp(
                x,
                context.x_axis[::-1],
                np.arange(context.x_axis.size - 1, -1, -1, dtype=np.float64),
                left=float(context.x_axis.size - 1),
                right=0.0,
            )
        )

    if context.y_axis[0] <= context.y_axis[-1]:
        row_f = float(
            np.interp(
                y,
                context.y_axis,
                np.arange(context.y_axis.size, dtype=np.float64),
                left=0.0,
                right=float(context.y_axis.size - 1),
            )
        )
    else:
        row_f = float(
            np.interp(
                y,
                context.y_axis[::-1],
                np.arange(context.y_axis.size - 1, -1, -1, dtype=np.float64),
                left=float(context.y_axis.size - 1),
                right=0.0,
            )
        )

    row0 = int(np.floor(row_f))
    col0 = int(np.floor(col_f))
    row1 = min(row0 + 1, context.z_grid.shape[0] - 1)
    col1 = min(col0 + 1, context.z_grid.shape[1] - 1)
    row_t = row_f - row0
    col_t = col_f - col0

    z00 = float(context.z_grid[row0, col0])
    z01 = float(context.z_grid[row0, col1])
    z10 = float(context.z_grid[row1, col0])
    z11 = float(context.z_grid[row1, col1])

    top = z00 * (1.0 - col_t) + z01 * col_t
    bottom = z10 * (1.0 - col_t) + z11 * col_t
    return float(top * (1.0 - row_t) + bottom * row_t)


def sample_geometry_elevation(
    context: TerrainContext,
    geometry: BaseGeometry,
    mode: str = "mean",
) -> float:
    points = geometry_sample_coordinates(geometry)
    if not points:
        representative = geometry.representative_point()
        points = [(representative.x, representative.y)]

    samples = np.array([sample_elevation(context, x, y) for x, y in points], dtype=np.float64)
    if mode == "min":
        return float(np.min(samples))
    if mode == "max":
        return float(np.max(samples))
    return float(np.mean(samples))


def _axis_spacing(axis: np.ndarray) -> float:
    if axis.size <= 1:
        return 1.0
    diffs = np.abs(np.diff(axis.astype(np.float64)))
    diffs = diffs[diffs > 1e-9]
    if diffs.size == 0:
        return 1.0
    return float(np.median(diffs))


def _densify_ring(coords: list[tuple[float, float]], max_segment_length: float) -> list[tuple[float, float]]:
    if len(coords) < 2:
        return coords
    dense: list[tuple[float, float]] = []
    max_segment_length = max(float(max_segment_length), 1.0)
    for index in range(len(coords) - 1):
        x1, y1 = coords[index]
        x2, y2 = coords[index + 1]
        dense.append((float(x1), float(y1)))
        segment_length = float(np.hypot(x2 - x1, y2 - y1))
        steps = max(1, int(np.ceil(segment_length / max_segment_length)))
        for step in range(1, steps):
            t = step / steps
            dense.append((float(x1 + (x2 - x1) * t), float(y1 + (y2 - y1) * t)))
    dense.append((float(coords[-1][0]), float(coords[-1][1])))
    return dense


def _iter_polygon_rings(geometry: BaseGeometry) -> list[list[tuple[float, float]]]:
    rings: list[list[tuple[float, float]]] = []
    if geometry.geom_type == "Polygon":
        rings.append(list(geometry.exterior.coords))
        rings.extend(list(interior.coords) for interior in geometry.interiors)
        return rings
    if geometry.geom_type == "MultiPolygon":
        for polygon in geometry.geoms:
            rings.extend(_iter_polygon_rings(polygon))
    return rings


def build_terrain_mesh(context: TerrainContext, base_height: float) -> trimesh.Trimesh:
    if Delaunay is None:
        safe_exit("Terrain icin scipy.spatial.Delaunay gerekli")

    rows, cols = context.z_grid.shape
    boundary = context.boundary_polygon_shifted
    prepared_boundary = prep(boundary)
    spacing_x = _axis_spacing(context.x_axis)
    spacing_y = _axis_spacing(context.y_axis)
    boundary_segment_length = max(min(spacing_x, spacing_y) * 0.8, 1.0)
    sample_stride = max(1, int(np.ceil(np.sqrt((rows * cols) / MAX_TERRAIN_TRIANGULATION_POINTS))))

    point_lookup: dict[tuple[float, float], int] = {}
    point_coords: list[tuple[float, float]] = []
    point_heights: list[float] = []

    def add_point(x: float, y: float, z: float | None = None) -> int:
        key = (round(float(x), 5), round(float(y), 5))
        existing = point_lookup.get(key)
        if existing is not None:
            return existing
        index = len(point_coords)
        point_lookup[key] = index
        point_coords.append((float(x), float(y)))
        point_heights.append(float(sample_elevation(context, x, y) if z is None else z))
        return index

    for row in range(0, rows, sample_stride):
        for col in range(0, cols, sample_stride):
            x = float(context.x_grid[row, col])
            y = float(context.y_grid[row, col])
            if prepared_boundary.covers(Point(x, y)):
                add_point(x, y, float(context.z_grid[row, col]))

    representative = boundary.representative_point()
    add_point(float(representative.x), float(representative.y))

    ring_indices: list[list[int]] = []
    for ring_coords in _iter_polygon_rings(boundary):
        dense_ring = _densify_ring(ring_coords, boundary_segment_length)
        indices: list[int] = []
        for x, y in dense_ring[:-1]:
            indices.append(add_point(x, y))
        if len(indices) >= 3:
            ring_indices.append(indices)

    if len(point_coords) < 3:
        safe_exit("Terrain mesh icin uygun hucre bulunamadi")

    points_2d = np.asarray(point_coords, dtype=np.float64)
    delaunay = Delaunay(points_2d)
    faces_top: list[list[int]] = []
    for simplex in delaunay.simplices:
        triangle = Polygon(points_2d[np.asarray(simplex, dtype=np.int64)])
        if triangle.area <= 1e-8:
            continue
        if prepared_boundary.covers(triangle):
            faces_top.append([int(simplex[0]), int(simplex[1]), int(simplex[2])])

    if not faces_top:
        safe_exit("Terrain triangulation basarisiz oldu")

    vertices_top = np.column_stack(
        [
            points_2d[:, 0],
            points_2d[:, 1],
            np.asarray(point_heights, dtype=np.float64),
        ]
    )
    faces_top_arr = np.asarray(faces_top, dtype=np.int64)
    min_z = float(np.min(vertices_top[:, 2]))
    bottom_z = min_z - float(base_height)

    vertices_bottom = vertices_top.copy()
    vertices_bottom[:, 2] = bottom_z
    vertex_count = vertices_top.shape[0]

    side_faces: list[list[int]] = []
    for indices in ring_indices:
        for ring_index, start in enumerate(indices):
            end = indices[(ring_index + 1) % len(indices)]
            side_faces.append([start, end, end + vertex_count])
            side_faces.append([start, end + vertex_count, start + vertex_count])

    faces_bottom_arr = faces_top_arr[:, ::-1] + vertex_count
    faces_all = np.vstack(
        [
            faces_top_arr,
            faces_bottom_arr,
            np.asarray(side_faces, dtype=np.int64),
        ]
    )
    vertices_all = np.vstack([vertices_top, vertices_bottom])
    solid_mesh = trimesh.Trimesh(vertices=vertices_all, faces=faces_all, process=False)
    return repair_mesh(solid_mesh)
