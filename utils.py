from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable

import geopandas as gpd
import numpy as np
import osmnx as ox
import trimesh
from shapely.affinity import translate
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
    box,
)
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union


ROAD_WIDTHS_M = {
    "motorway": 18.0,
    "trunk": 16.0,
    "primary": 12.0,
    "secondary": 10.0,
    "tertiary": 8.0,
    "unclassified": 7.0,
    "residential": 6.0,
    "living_street": 5.0,
    "service": 4.0,
    "pedestrian": 4.0,
    "footway": 2.0,
    "path": 1.6,
    "steps": 1.5,
    "cycleway": 1.8,
    "track": 3.0,
}

RAILWAY_WIDTHS_M = {
    "rail": 4.2,
    "narrow_gauge": 3.4,
    "light_rail": 3.6,
    "subway": 3.4,
    "tram": 2.8,
    "monorail": 2.8,
    "funicular": 2.6,
    "preserved": 3.3,
}

WATERWAY_WIDTHS_M = {
    "river": 18.0,
    "canal": 10.0,
    "stream": 4.0,
    "drain": 2.0,
    "ditch": 2.0,
}

BUILDING_DEFAULT_HEIGHTS_M = {
    "apartments": 22.0,
    "commercial": 18.0,
    "garage": 4.0,
    "hospital": 18.0,
    "house": 10.0,
    "industrial": 14.0,
    "mosque": 25.0,
    "office": 18.0,
    "residential": 14.0,
    "school": 14.0,
    "warehouse": 12.0,
    "yes": 12.0,
}


def safe_exit(message: str) -> None:
    raise SystemExit(message)


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def sanitize_filename(text: str) -> str:
    text = text.strip().replace(",", "_").replace(" ", "_")
    text = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_") or "model"


def make_cache_key(payload: Any) -> str:
    return hashlib.sha1(
        json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
    ).hexdigest()


def get_cache_key(payload: Any) -> str:
    return make_cache_key(payload)


def _gdf_to_cache_payload(gdf: gpd.GeoDataFrame) -> dict[str, Any]:
    return {
        "crs": gdf.crs.to_string() if gdf.crs else "EPSG:4326",
        "features": gdf.__geo_interface__["features"],
    }


def save_gdf_cache(path: str | Path, gdf: gpd.GeoDataFrame) -> None:
    cache_path = Path(path)
    ensure_directory(cache_path.parent)
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(_gdf_to_cache_payload(gdf), handle, ensure_ascii=False, default=str)


def load_gdf_cache(path: str | Path) -> gpd.GeoDataFrame | None:
    cache_path = Path(path)
    if not cache_path.exists():
        return None

    with cache_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    features = payload.get("features", [])
    crs = payload.get("crs") or "EPSG:4326"
    if not features:
        return empty_gdf(crs)
    gdf = gpd.GeoDataFrame.from_features(features, crs=crs)
    if "geometry" not in gdf.columns:
        gdf = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=crs)
    return gdf


def cache_osm_data(cache_dir: str | Path, key: str, data: Any) -> None:
    cache_path = ensure_directory(cache_dir) / f"{key}.json"
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, default=str)


def load_cached_osm_data(cache_dir: str | Path, key: str) -> Any | None:
    cache_path = Path(cache_dir) / f"{key}.json"
    if not cache_path.exists():
        return None

    with cache_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def cache_data(data: Any, key: str, cache_dir: str | Path) -> None:
    if isinstance(data, gpd.GeoDataFrame):
        save_gdf_cache(Path(cache_dir) / f"{key}.json", data)
        return

    cache_path = ensure_directory(cache_dir) / f"{key}.json"
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, default=str)


def load_cached_data(key: str, cache_dir: str | Path) -> Any | None:
    cache_path = Path(cache_dir) / f"{key}.json"
    if not cache_path.exists():
        return None

    try:
        return load_gdf_cache(cache_path)
    except Exception:
        with cache_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)


def empty_gdf(crs: str | None = "EPSG:4326") -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=crs)


def normalize_tag_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        for item in value:
            normalized = normalize_tag_value(item)
            if normalized:
                return normalized
        return None

    text = str(value).strip().lower()
    if not text or text == "nan":
        return None
    if ";" in text:
        text = text.split(";", 1)[0].strip()
    return text or None


_FIRST_NUMBER_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?")


def parse_first_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        if np.isnan(value):
            return None
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    match = _FIRST_NUMBER_RE.search(text)
    if not match:
        return None

    try:
        return float(match.group(0).replace(",", "."))
    except ValueError:
        return None


def parse_length_to_meters(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        if np.isnan(value):
            return None
        return float(value)

    text = str(value).strip().lower()
    if not text:
        return None

    numeric = parse_first_float(text)
    if numeric is None:
        return None

    if "ft" in text or "feet" in text or "'" in text:
        return numeric * 0.3048
    if "cm" in text:
        return numeric / 100.0
    if "mm" in text:
        return numeric / 1000.0
    return numeric


def building_default_height(row: Any, fallback: float) -> float:
    building_type = normalize_tag_value(row.get("building"))
    if not building_type:
        return fallback
    return max(fallback, BUILDING_DEFAULT_HEIGHTS_M.get(building_type, fallback))


def get_height(
    row: Any,
    levels_multiplier: float = 3.2,
    default_height: float = 12.0,
    min_height: float = 0.0,
) -> float:
    explicit_height = parse_length_to_meters(row.get("height"))
    if explicit_height and explicit_height > 0:
        roof_height = parse_length_to_meters(row.get("roof:height")) or 0.0
        return max(explicit_height + roof_height, min_height)

    levels = parse_first_float(row.get("building:levels"))
    roof_levels = parse_first_float(row.get("roof:levels")) or 0.0
    if levels and levels > 0:
        computed_height = levels * levels_multiplier + roof_levels * (levels_multiplier * 0.6)
        return max(computed_height, min_height)

    building_height = building_default_height(row, default_height)
    return max(building_height, min_height)


def choose_buildings(
    buildings: gpd.GeoDataFrame,
    selection_mode: str,
    max_buildings: int | None,
) -> gpd.GeoDataFrame:
    if buildings.empty:
        return buildings

    if selection_mode == "largest":
        buildings = buildings.sort_values(by="footprint_area", ascending=False)
    elif selection_mode == "random":
        buildings = buildings.sample(frac=1, random_state=42)
    elif selection_mode == "all":
        pass
    else:
        safe_exit(f"Gecersiz secim modu: {selection_mode}")

    if max_buildings is not None:
        buildings = buildings.head(min(max_buildings, len(buildings)))

    return buildings


def clean_geometry(geometry: BaseGeometry | None) -> BaseGeometry | None:
    if geometry is None:
        return None
    if geometry.is_empty:
        return None
    if not geometry.is_valid:
        geometry = geometry.buffer(0)
    if geometry.is_empty:
        return None
    return geometry


def clip_geodataframe(gdf: gpd.GeoDataFrame, polygon: BaseGeometry) -> gpd.GeoDataFrame:
    if gdf is None or gdf.empty:
        return empty_gdf(gdf.crs if gdf is not None else "EPSG:4326")

    clipped = gdf.copy()
    clipped = clipped[clipped.geometry.notnull()].copy()
    clipped["geometry"] = clipped.geometry.apply(
        lambda geom: clean_geometry(geom.intersection(polygon)) if geom is not None else None
    )
    clipped = clipped[clipped.geometry.notnull()].copy()
    return clipped


def project_geodataframe(gdf: gpd.GeoDataFrame, target_crs: Any) -> gpd.GeoDataFrame:
    if gdf.empty:
        return empty_gdf(target_crs)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf.to_crs(target_crs)


def translate_geodataframe(
    gdf: gpd.GeoDataFrame,
    x_offset: float,
    y_offset: float,
) -> gpd.GeoDataFrame:
    translated = gdf.copy()
    translated["geometry"] = translated.geometry.apply(
        lambda geom: translate(geom, xoff=x_offset, yoff=y_offset) if geom is not None else None
    )
    return translated


def thicken_small_geometry(geometry: BaseGeometry, buffer_m: float = 0.0) -> BaseGeometry:
    if buffer_m <= 0:
        return geometry
    geometry = clean_geometry(geometry)
    if geometry is None:
        return geometry

    try:
        return geometry.buffer(buffer_m).buffer(-buffer_m * 0.2)
    except Exception:
        return geometry


def iter_polygons(geometry: BaseGeometry | None) -> Iterable[Polygon]:
    geometry = clean_geometry(geometry)
    if geometry is None:
        return
    if isinstance(geometry, Polygon):
        yield geometry
        return
    if isinstance(geometry, MultiPolygon):
        for polygon in geometry.geoms:
            yield from iter_polygons(polygon)
        return
    if isinstance(geometry, GeometryCollection):
        for child in geometry.geoms:
            yield from iter_polygons(child)


def iter_lines(geometry: BaseGeometry | None) -> Iterable[LineString]:
    geometry = clean_geometry(geometry)
    if geometry is None:
        return
    if isinstance(geometry, LineString):
        yield geometry
        return
    if isinstance(geometry, MultiLineString):
        for line in geometry.geoms:
            yield from iter_lines(line)
        return
    if isinstance(geometry, GeometryCollection):
        for child in geometry.geoms:
            yield from iter_lines(child)


def road_width_from_highway(value: Any) -> float:
    normalized = normalize_tag_value(value)
    return ROAD_WIDTHS_M.get(normalized or "", 5.0)


def railway_width_from_type(value: Any) -> float:
    normalized = normalize_tag_value(value)
    return RAILWAY_WIDTHS_M.get(normalized or "", 3.6)


def water_width_from_waterway(value: Any) -> float:
    normalized = normalize_tag_value(value)
    return WATERWAY_WIDTHS_M.get(normalized or "", 3.0)


def parse_boolish(value: Any) -> bool | None:
    normalized = normalize_tag_value(value)
    if normalized in {"yes", "true", "1"}:
        return True
    if normalized in {"no", "false", "0"}:
        return False
    return None


def parse_lane_count(value: Any) -> float | None:
    if value is None:
        return None

    if isinstance(value, (int, float, np.integer, np.floating)):
        if np.isnan(value):
            return None
        return float(value)

    text = str(value).strip().lower()
    if not text:
        return None

    separators = []
    if "|" in text:
        separators.append("|")
    if ";" in text:
        separators.append(";")
    if "/" in text:
        separators.append("/")

    for separator in separators:
        text = text.replace(separator, ";")

    parts = [parse_first_float(part) for part in text.split(";")]
    parts = [part for part in parts if part is not None and part > 0]
    if not parts:
        return None

    if "|" in str(value):
        return float(sum(parts))
    return float(max(parts))


def estimate_lane_width_m(highway_value: Any) -> float:
    highway = normalize_tag_value(highway_value)
    if highway in {"motorway", "trunk"}:
        return 3.7
    if highway in {"primary", "secondary"}:
        return 3.4
    if highway in {"tertiary", "unclassified", "residential"}:
        return 3.1
    if highway in {"living_street", "service", "track"}:
        return 2.8
    if highway in {"pedestrian", "footway", "path", "steps", "cycleway"}:
        return 1.8
    return 3.0


def estimate_road_surface_width(row: Any, fallback_width_m: float = 0.0) -> float:
    explicit_width = parse_length_to_meters(row.get("width"))
    lanes = parse_lane_count(row.get("lanes"))
    lane_width = estimate_lane_width_m(row.get("highway"))
    lanes_width = lanes * lane_width if lanes else None
    default_width = road_width_from_highway(row.get("highway"))

    candidates = [fallback_width_m, default_width]
    if explicit_width is not None:
        candidates.append(explicit_width)
    if lanes_width is not None:
        candidates.append(lanes_width)

    return max(candidate for candidate in candidates if candidate is not None and candidate > 0)


def parse_railway_gauge_m(value: Any) -> float | None:
    gauge_value = parse_first_float(value)
    if gauge_value is None or gauge_value <= 0:
        return None

    text = str(value).strip().lower()
    if "mm" in text:
        return gauge_value / 1000.0
    if "cm" in text:
        return gauge_value / 100.0
    if "ft" in text or "feet" in text or "'" in text:
        return gauge_value * 0.3048
    if re.search(r"\bm\b", text):
        return gauge_value

    # OSM'de gauge degeri cogunlukla milimetre olarak islenir.
    if gauge_value >= 20:
        return gauge_value / 1000.0
    return gauge_value


def estimate_railway_surface_width(row: Any, fallback_width_m: float = 0.0) -> float:
    explicit_width = parse_length_to_meters(row.get("width"))
    default_single_track_width = railway_width_from_type(row.get("railway"))
    gauge_m = parse_railway_gauge_m(row.get("gauge"))
    if gauge_m is not None:
        default_single_track_width = max(default_single_track_width, gauge_m + 1.0)

    track_count = parse_first_float(row.get("tracks"))
    if track_count is None or track_count <= 0:
        track_count = 1.0
    track_count = float(np.clip(track_count, 1.0, 6.0))

    tracks_total_width = default_single_track_width * track_count
    if track_count > 1.0:
        tracks_total_width += (track_count - 1.0) * 0.7

    candidates = [fallback_width_m, default_single_track_width, tracks_total_width]
    if explicit_width is not None:
        candidates.append(explicit_width)

    return max(candidate for candidate in candidates if candidate is not None and candidate > 0)


def estimate_visible_transport_layer(row: Any) -> float | None:
    layer_value = parse_first_float(row.get("layer")) or 0.0
    bridge = parse_boolish(row.get("bridge"))
    tunnel = parse_boolish(row.get("tunnel"))
    covered = parse_boolish(row.get("covered"))

    if tunnel is True or covered is True:
        return None
    if layer_value < 0:
        return None
    if bridge is True and layer_value < 1:
        layer_value = 1.0
    return float(layer_value)


def estimate_visible_road_layer(row: Any) -> float | None:
    return estimate_visible_transport_layer(row)


def should_split_dual_carriageway(row: Any, total_width_m: float) -> bool:
    if total_width_m < 10.5:
        return False

    if parse_boolish(row.get("oneway")) is True:
        return False

    lanes = parse_lane_count(row.get("lanes")) or 0.0
    highway = normalize_tag_value(row.get("highway"))

    if lanes >= 4:
        return True
    if highway in {"motorway", "trunk", "primary"} and total_width_m >= 13.0:
        return True
    return False


def offset_line_geometry(line: LineString, distance_m: float) -> BaseGeometry | None:
    try:
        if hasattr(line, "offset_curve"):
            offset = line.offset_curve(distance_m, quad_segs=8, join_style=1, mitre_limit=2.0)
        else:
            side = "left" if distance_m >= 0 else "right"
            offset = line.parallel_offset(
                abs(distance_m),
                side=side,
                resolution=8,
                join_style=1,
                mitre_limit=2.0,
            )
    except Exception:
        return None

    return clean_geometry(offset)


def geometry_to_polygonal_feature(
    geometry: BaseGeometry,
    width_m: float,
    clip_polygon: BaseGeometry | None = None,
) -> BaseGeometry | None:
    geometry = clean_geometry(geometry)
    if geometry is None:
        return None

    if geometry.geom_type in {"Polygon", "MultiPolygon"}:
        polygonal = geometry
    else:
        polygonal = geometry.buffer(max(width_m / 2.0, 0.1), cap_style=1, join_style=1, quad_segs=8)

    polygonal = clean_geometry(polygonal)
    if polygonal is None:
        return None

    if clip_polygon is not None:
        polygonal = clean_geometry(polygonal.intersection(clip_polygon))

    return polygonal


def build_dual_carriageway_geometry(
    geometry: BaseGeometry,
    total_width_m: float,
    clip_polygon: BaseGeometry | None = None,
) -> BaseGeometry | None:
    geometry = clean_geometry(geometry)
    if geometry is None:
        return None

    if geometry.geom_type in {"Polygon", "MultiPolygon"}:
        return geometry_to_polygonal_feature(geometry, total_width_m, clip_polygon=clip_polygon)

    median_width = min(max(total_width_m * 0.16, 1.2), 4.5)
    carriage_width = (total_width_m - median_width) / 2.0
    if carriage_width <= 1.6:
        return geometry_to_polygonal_feature(geometry, total_width_m, clip_polygon=clip_polygon)

    offset_distance = carriage_width / 2.0 + median_width / 2.0
    parts: list[BaseGeometry] = []

    for line in iter_lines(geometry):
        if line.length < total_width_m * 1.25:
            polygonal = geometry_to_polygonal_feature(line, total_width_m, clip_polygon=None)
            if polygonal is not None:
                parts.append(polygonal)
            continue

        left_offset = offset_line_geometry(line, offset_distance)
        right_offset = offset_line_geometry(line, -offset_distance)

        if left_offset is None or right_offset is None:
            polygonal = geometry_to_polygonal_feature(line, total_width_m, clip_polygon=None)
            if polygonal is not None:
                parts.append(polygonal)
            continue

        left_surface = geometry_to_polygonal_feature(left_offset, carriage_width, clip_polygon=None)
        right_surface = geometry_to_polygonal_feature(right_offset, carriage_width, clip_polygon=None)

        if left_surface is not None:
            parts.append(left_surface)
        if right_surface is not None:
            parts.append(right_surface)

    if not parts:
        return None

    merged = clean_geometry(unary_union(parts))
    if merged is None:
        return None

    if clip_polygon is not None:
        merged = clean_geometry(merged.intersection(clip_polygon))

    return merged


def merge_polygon_geometries(
    geometries: Iterable[BaseGeometry],
    min_area_m2: float = 0.0,
) -> list[Polygon]:
    cleaned = [clean_geometry(geometry) for geometry in geometries]
    cleaned = [geometry for geometry in cleaned if geometry is not None]
    if not cleaned:
        return []

    merged = clean_geometry(unary_union(cleaned))
    if merged is None:
        return []

    polygons = []
    for polygon in iter_polygons(merged):
        if polygon.area >= min_area_m2:
            polygons.append(polygon)
    return polygons


def split_polygon_geometry(
    geometry: BaseGeometry,
    max_chunk_size_m: float,
    min_area_m2: float = 0.0,
) -> list[Polygon]:
    geometry = clean_geometry(geometry)
    if geometry is None:
        return []

    max_chunk_size_m = max(float(max_chunk_size_m), 1.0)
    pending = list(iter_polygons(geometry))
    pieces: list[Polygon] = []

    while pending:
        polygon = pending.pop()
        minx, miny, maxx, maxy = polygon.bounds
        width = maxx - minx
        height = maxy - miny

        if max(width, height) <= max_chunk_size_m * 1.1:
            if polygon.area >= min_area_m2:
                pieces.append(polygon)
            continue

        if width >= height:
            midpoint = (minx + maxx) / 2.0
            cutters = [
                box(minx - 0.01, miny - 0.01, midpoint, maxy + 0.01),
                box(midpoint, miny - 0.01, maxx + 0.01, maxy + 0.01),
            ]
        else:
            midpoint = (miny + maxy) / 2.0
            cutters = [
                box(minx - 0.01, miny - 0.01, maxx + 0.01, midpoint),
                box(minx - 0.01, midpoint, maxx + 0.01, maxy + 0.01),
            ]

        child_polygons: list[Polygon] = []
        for cutter in cutters:
            part = clean_geometry(polygon.intersection(cutter))
            if part is None:
                continue
            child_polygons.extend(iter_polygons(part))

        if len(child_polygons) <= 1:
            if polygon.area >= min_area_m2:
                pieces.append(polygon)
            continue

        pending.extend(child_polygons)

    return [polygon for polygon in pieces if polygon.area >= min_area_m2]


def geometry_dimensions(geometry: BaseGeometry) -> tuple[float, float]:
    minx, miny, maxx, maxy = geometry.bounds
    return float(maxx - minx), float(maxy - miny)


def _find_vertex_index(
    vertices_2d: np.ndarray,
    x: float,
    y: float,
    tolerance: float = 1e-5,
) -> int | None:
    if vertices_2d.size == 0:
        return None

    deltas = vertices_2d - np.array([[float(x), float(y)]], dtype=np.float64)
    distances_sq = np.einsum("ij,ij->i", deltas, deltas)
    index = int(np.argmin(distances_sq))
    if float(distances_sq[index]) <= float(tolerance * tolerance):
        return index
    return None


def _ring_side_faces(vertices_2d: np.ndarray, ring_coords: list[tuple[float, float]], top_index_offset: int) -> list[list[int]]:
    side_faces: list[list[int]] = []
    if len(ring_coords) < 2:
        return side_faces

    for start, end in zip(ring_coords, ring_coords[1:]):
        bottom_start = _find_vertex_index(vertices_2d, start[0], start[1])
        bottom_end = _find_vertex_index(vertices_2d, end[0], end[1])
        if bottom_start is None or bottom_end is None or bottom_start == bottom_end:
            continue

        top_start = bottom_start + top_index_offset
        top_end = bottom_end + top_index_offset
        side_faces.append([bottom_start, bottom_end, top_end])
        side_faces.append([bottom_start, top_end, top_start])

    return side_faces


def create_surface_following_prism_meshes(
    geometry: BaseGeometry,
    bottom_surface_z_resolver: Callable[[float, float], float],
    top_surface_z_resolver: Callable[[float, float], float],
    min_area_m2: float = 0.0,
) -> list[trimesh.Trimesh]:
    meshes: list[trimesh.Trimesh] = []
    geometry = clean_geometry(geometry)
    if geometry is None:
        return meshes

    for polygon in iter_polygons(geometry):
        if polygon.area < min_area_m2:
            continue

        try:
            vertices_2d, faces = trimesh.creation.triangulate_polygon(polygon)
        except Exception as exc:
            print(f"Yuzeyi takip eden mesh olusturma hatasi: {exc}")
            continue

        vertices_2d = np.asarray(vertices_2d, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64)
        if vertices_2d.size == 0 or faces.size == 0:
            continue

        bottom_z = np.array(
            [float(bottom_surface_z_resolver(float(x), float(y))) for x, y in vertices_2d],
            dtype=np.float64,
        )
        top_z = np.array(
            [float(top_surface_z_resolver(float(x), float(y))) for x, y in vertices_2d],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(bottom_z)) or not np.all(np.isfinite(top_z)):
            continue

        top_index_offset = len(vertices_2d)
        bottom_vertices = np.column_stack([vertices_2d, bottom_z])
        top_vertices = np.column_stack([vertices_2d, top_z])
        vertices_3d = np.vstack([bottom_vertices, top_vertices])

        side_faces: list[list[int]] = []
        side_faces.extend(
            _ring_side_faces(vertices_2d, list(polygon.exterior.coords), top_index_offset)
        )
        for interior in polygon.interiors:
            side_faces.extend(_ring_side_faces(vertices_2d, list(interior.coords), top_index_offset))

        mesh_faces = [faces[:, ::-1], faces + top_index_offset]
        if side_faces:
            mesh_faces.append(np.asarray(side_faces, dtype=np.int64))

        mesh = trimesh.Trimesh(vertices=vertices_3d, faces=np.vstack(mesh_faces), process=False)
        meshes.append(repair_mesh(mesh))

    return meshes


def extrude_geometry_at_base(
    geometry: BaseGeometry,
    height_m: float,
    base_z: float,
    min_area_m2: float = 0.0,
) -> list[trimesh.Trimesh]:
    meshes: list[trimesh.Trimesh] = []
    for polygon in iter_polygons(geometry):
        if polygon.area < min_area_m2:
            continue
        try:
            mesh = trimesh.creation.extrude_polygon(polygon, float(height_m))
            mesh.apply_translation([0.0, 0.0, float(base_z)])
            meshes.append(mesh)
        except Exception as exc:
            print(f"Mesh olusturma hatasi: {exc}")
    return meshes


def sample_points_in_polygon_grid(
    geometry: BaseGeometry,
    spacing_m: float,
    margin_m: float = 0.0,
    max_points: int | None = None,
) -> list[tuple[float, float]]:
    spacing_m = max(float(spacing_m), 1.0)
    points: list[tuple[float, float]] = []

    for polygon in iter_polygons(geometry):
        sample_polygon = clean_geometry(polygon.buffer(-margin_m, join_style=1)) if margin_m > 0 else polygon
        if sample_polygon is None:
            sample_polygon = polygon

        for inner_polygon in iter_polygons(sample_polygon):
            minx, miny, maxx, maxy = inner_polygon.bounds
            center = inner_polygon.representative_point()
            phase_x = abs(center.x * 0.61803398875) % spacing_m
            phase_y = abs(center.y * 0.41421356237) % spacing_m
            start_x = minx + min(max(phase_x, spacing_m * 0.25), spacing_m * 0.75)
            start_y = miny + min(max(phase_y, spacing_m * 0.25), spacing_m * 0.75)

            x = start_x
            while x < maxx:
                y = start_y
                while y < maxy:
                    point = Point(x, y)
                    if inner_polygon.contains(point):
                        points.append((x, y))
                        if max_points is not None and len(points) >= max_points:
                            return points
                    y += spacing_m
                x += spacing_m

            if not points:
                representative = inner_polygon.representative_point()
                points.append((representative.x, representative.y))
                if max_points is not None and len(points) >= max_points:
                    return points

    return points[:max_points] if max_points is not None else points


def create_tree_mesh(
    x: float,
    y: float,
    base_z: float,
    trunk_radius_m: float,
    trunk_height_m: float,
    canopy_radius_m: float,
    canopy_height_m: float,
    canopy_style: str = "cone",
) -> trimesh.Trimesh:
    trunk = trimesh.creation.cylinder(radius=trunk_radius_m, height=trunk_height_m, sections=8)
    trunk.apply_translation([x, y, base_z + trunk_height_m / 2.0])

    if canopy_style == "sphere":
        canopy = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
        canopy.apply_scale([canopy_radius_m, canopy_radius_m, canopy_height_m / 2.0])
        canopy.apply_translation([x, y, base_z + trunk_height_m + canopy_height_m / 2.0])
    else:
        canopy = trimesh.creation.cone(radius=canopy_radius_m, height=canopy_height_m, sections=9)
        canopy.apply_translation([x, y, base_z + trunk_height_m])

    return trimesh.util.concatenate([trunk, canopy])


def geometry_sample_coordinates(geometry: BaseGeometry) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    geometry = clean_geometry(geometry)
    if geometry is None:
        return points

    representative = geometry.representative_point()
    points.append((representative.x, representative.y))

    if isinstance(geometry, Polygon):
        coords = list(geometry.exterior.coords)
        step = max(1, len(coords) // 12)
        points.extend(coords[::step])
        return points

    if isinstance(geometry, MultiPolygon):
        for polygon in list(geometry.geoms)[:4]:
            points.extend(geometry_sample_coordinates(polygon))
        return points

    if isinstance(geometry, LineString):
        coords = list(geometry.coords)
        step = max(1, len(coords) // 8)
        points.extend(coords[::step])
        return points

    if isinstance(geometry, MultiLineString):
        for line in list(geometry.geoms)[:4]:
            points.extend(geometry_sample_coordinates(line))
        return points

    return points


def create_meshes_for_geometries(
    geometries: Iterable[BaseGeometry],
    heights_m: Iterable[float],
    elevation_sampler: Callable[[BaseGeometry, str], float] | None = None,
    base_mode: str = "mean",
    min_area_m2: float = 0.0,
) -> list[trimesh.Trimesh]:
    meshes: list[trimesh.Trimesh] = []

    for geometry, height_m in zip(geometries, heights_m):
        if height_m is None or height_m <= 0:
            continue

        geometry = clean_geometry(geometry)
        if geometry is None:
            continue

        base_z = elevation_sampler(geometry, base_mode) if elevation_sampler else 0.0

        for polygon in iter_polygons(geometry):
            if polygon.area < min_area_m2:
                continue
            try:
                mesh = trimesh.creation.extrude_polygon(polygon, float(height_m))
                mesh.apply_translation([0.0, 0.0, float(base_z)])
                meshes.append(mesh)
            except Exception as exc:
                print(f"Mesh olusturma hatasi: {exc}")

    return meshes


def create_building_meshes(
    buildings_gdf: gpd.GeoDataFrame,
    dem: np.ndarray | None = None,
    dem_bounds: tuple[float, float, float, float] | None = None,
    utm_crs: Any | None = None,
    levels_multiplier: float = 3.0,
    default_height: float = 10.0,
    buffer_m: float = 0.0,
) -> list[trimesh.Trimesh]:
    if buildings_gdf.empty:
        return []

    if "height_m" not in buildings_gdf.columns:
        heights = buildings_gdf.apply(
            lambda row: get_height(row, levels_multiplier, default_height),
            axis=1,
        )
    else:
        heights = buildings_gdf["height_m"]

    meshes: list[trimesh.Trimesh] = []

    for geometry, height in zip(buildings_gdf.geometry, heights):
        geometry = clean_geometry(geometry)
        if geometry is None:
            continue

        if buffer_m > 0:
            geometry = thicken_small_geometry(geometry, buffer_m)

        base_z = 0.0
        if dem is not None and dem_bounds is not None:
            centroid = geometry.representative_point()
            if utm_crs is not None:
                from pyproj import Transformer

                transformer = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
                lon, lat = transformer.transform(centroid.x, centroid.y)
            else:
                lon, lat = centroid.x, centroid.y

            west, south, east, north = dem_bounds
            rows, cols = dem.shape
            if east > west and north > south:
                col = int(np.clip((lon - west) / (east - west) * (cols - 1), 0, cols - 1))
                row = int(np.clip((north - lat) / (north - south) * (rows - 1), 0, rows - 1))
                base_z = float(dem[row, col])

        for polygon in iter_polygons(geometry):
            try:
                mesh = trimesh.creation.extrude_polygon(polygon, float(height))
                mesh.apply_translation([0.0, 0.0, float(base_z)])
                meshes.append(mesh)
            except Exception as exc:
                print(f"Mesh olusturma hatasi: {exc}")

    return meshes


def repair_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    try:
        mesh.merge_vertices()
    except Exception:
        pass
    try:
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass
    try:
        trimesh.repair.fix_normals(mesh)
    except Exception:
        pass
    try:
        trimesh.repair.fill_holes(mesh)
    except Exception:
        pass
    return mesh


def get_place_center(place: str) -> tuple[float, float]:
    try:
        place_gdf = ox.geocode_to_gdf(place)
        if len(place_gdf) > 0:
            centroid = place_gdf.to_crs(place_gdf.estimate_utm_crs()).geometry.centroid.to_crs("EPSG:4326")
            return float(centroid.y.iloc[0]), float(centroid.x.iloc[0])
    except Exception:
        pass

    try:
        lat, lon = ox.geocode(place)
        return float(lat), float(lon)
    except Exception as exc:
        safe_exit(f"Konum cozumlenemedi: {place} ({exc})")


def get_place_boundary_polygon(place: str) -> BaseGeometry:
    try:
        place_gdf = ox.geocode_to_gdf(place)
    except Exception as exc:
        safe_exit(f"Yer siniri cozumlenemedi: {place} ({exc})")

    if place_gdf.empty:
        safe_exit(f"Yer siniri bulunamadi: {place}")

    place_gdf = place_gdf[place_gdf.geometry.notnull()].copy()
    if place_gdf.empty:
        safe_exit(f"Yer siniri geometrisi bulunamadi: {place}")

    geometry: BaseGeometry | None = None
    try:
        projected = place_gdf.to_crs(place_gdf.estimate_utm_crs())
        largest_index = projected.geometry.area.idxmax()
        geometry = clean_geometry(place_gdf.loc[largest_index].geometry)
    except Exception:
        geometry = clean_geometry(place_gdf.geometry.iloc[0])

    if geometry is None:
        safe_exit(f"Yer siniri geometrisi gecersiz: {place}")

    if geometry.geom_type not in {"Polygon", "MultiPolygon"}:
        safe_exit(
            f"Yer siniri polygon degil: {place}. Bu konum icin place-radius modu kullanman daha uygun olabilir."
        )

    return geometry


def make_circle_polygon(lat: float, lon: float, radius_m: float) -> Polygon:
    point_gdf = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326")
    utm_crs = point_gdf.estimate_utm_crs()
    polygon = point_gdf.to_crs(utm_crs).buffer(radius_m).to_crs("EPSG:4326").iloc[0]
    return polygon


def build_bbox_polygon(north: float, south: float, east: float, west: float) -> Polygon:
    if south > north:
        south, north = north, south
    if west > east:
        west, east = east, west
    if south == north or west == east:
        safe_exit("Gecersiz bbox: alan sifir olamaz")
    return box(west, south, east, north)


def parse_coordinate_polygon(text: str) -> Polygon:
    pairs = [pair.strip() for pair in text.split(";") if pair.strip()]
    coordinates: list[tuple[float, float]] = []

    for pair in pairs:
        if "," in pair:
            left, right = pair.split(",", 1)
        else:
            parts = pair.split()
            if len(parts) != 2:
                safe_exit("Koordinatlar 'lat,lon; lat,lon; ...' formatinda olmali")
            left, right = parts

        lat = parse_first_float(left)
        lon = parse_first_float(right)
        if lat is None or lon is None:
            safe_exit(f"Koordinat okunamadi: {pair}")
        coordinates.append((float(lon), float(lat)))

    if len(coordinates) < 3:
        safe_exit("Polygon icin en az 3 koordinat gerekli")

    polygon = Polygon(coordinates)
    polygon = clean_geometry(polygon)
    if polygon is None or polygon.area == 0:
        safe_exit("Polygon koordinatlari gecersiz")
    return polygon
