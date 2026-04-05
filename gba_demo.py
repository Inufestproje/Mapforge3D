from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import trimesh
from pyproj import Transformer
from shapely.affinity import scale as scale_geometry
from shapely.affinity import translate
from shapely.geometry import Point, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from main import (
    PARK_TAGS,
    PRINT_PROFILES,
    ROAD_SKIP_VALUES,
    ROAD_TAGS,
    WATER_LINE_TAGS,
    WATER_POLYGON_TAGS,
    build_buffered_linear_layer_meshes,
    build_park_layer_meshes,
    build_polygon_layer_meshes,
    configure_osmnx,
    create_building_detail_meshes,
    fetch_osm_layer,
    finalize_scene_mesh,
)
from terrain import build_terrain_context, build_terrain_mesh, sample_elevation, sample_geometry_elevation
from utils import (
    build_dual_carriageway_geometry,
    clean_geometry,
    create_meshes_for_geometries,
    create_surface_following_prism_meshes,
    create_tree_mesh,
    ensure_directory,
    estimate_road_surface_width,
    estimate_visible_road_layer,
    geometry_to_polygonal_feature,
    iter_lines,
    merge_polygon_geometries,
    normalize_tag_value,
    parse_length_to_meters,
    repair_mesh,
    sanitize_filename,
    thicken_small_geometry,
    translate_geodataframe,
    water_width_from_waterway,
)


HF_ODBL_ROOT = "https://huggingface.co/datasets/zhu-xlab/GBA.ODbLPolygon/resolve/main"
HF_LOD1_ROOT = "https://huggingface.co/datasets/zhu-xlab/GBA.LoD1/resolve/main"
TREE_POINT_TAGS = {"natural": ["tree"]}
TREE_ROW_TAGS = {"natural": ["tree_row"]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GlobalBuildingAtlas'tan kucuk bir alan indirip GeoJSON ve detayli STL sahnesi uretir.",
    )
    parser.add_argument("--region", default="southamerica", help="Tile bolgesi. Ornek: europe")
    parser.add_argument(
        "--tile",
        default="w035_n00_w030_s05",
        help="5x5 derece tile adi. Ornek: e000_n60_e005_n55",
    )
    parser.add_argument("--window-size-m", type=float, default=480.0, help="Ornek alan genisligi")
    parser.add_argument("--target-size-mm", type=float, default=120.0, help="STL hedef boyutu")
    parser.add_argument("--base-thickness-mm", type=float, default=6.0, help="Taban kalinligi")
    parser.add_argument(
        "--height-exaggeration",
        type=float,
        default=4.0,
        help="GBA bina yuksekliklerini daha gorunur yapmak icin carpani",
    )
    parser.add_argument(
        "--min-building-height-mm",
        type=float,
        default=None,
        help="Bina minimum yuksekligi. Bos birakilirsa profil varsayilani kullanilir",
    )
    parser.add_argument("--print-profile", choices=sorted(PRINT_PROFILES), default="fdm")
    parser.add_argument("--terrain-zoom", type=int, default=14)
    parser.add_argument("--terrain-max-size", type=int, default=280)
    parser.add_argument("--terrain-z-scale", type=float, default=1.0)
    parser.add_argument("--terrain-smoothing", type=float, default=0.15)
    parser.add_argument("--download-dir", default="cache/gba_demo", help="Indirme klasoru")
    parser.add_argument(
        "--sample-dir",
        default="cache/gba_demo/sample",
        help="Ara cikti ve ornek GeoJSON klasoru",
    )
    parser.add_argument("--stl-dir", default="3_boyutlu_stl_dosyalari", help="STL cikti klasoru")
    parser.add_argument("--scene-output", default=None, help="Detayli sahne STL yolu")
    parser.add_argument("--sample-center-lat", type=float, default=-3.8463, help="Ornek merkez lat")
    parser.add_argument("--sample-center-lon", type=float, default=-32.4119, help="Ornek merkez lon")
    parser.add_argument("--without-roads", action="store_true")
    parser.add_argument("--without-water", action="store_true")
    parser.add_argument("--without-parks", action="store_true")
    return parser


def dataset_urls(region: str, tile: str) -> dict[str, str]:
    return {
        "odbl": f"{HF_ODBL_ROOT}/{region}/{tile}.geojson?download=true",
        "polygon": f"{HF_LOD1_ROOT}/Polygon/{region}/{tile}.geojson?download=true",
        "lod1": f"{HF_LOD1_ROOT}/LoD1/{region}/{tile}.json?download=true",
    }


def download_file(url: str, destination: Path) -> Path:
    if destination.exists() and destination.stat().st_size > 0:
        return destination

    response = requests.get(url, timeout=300)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def load_height_lookup(json_path: Path) -> dict[str, dict[str, float | None]]:
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def drop_duplicate_polygon_buildings(
    odbl_gdf: gpd.GeoDataFrame,
    polygon_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    if odbl_gdf.empty or polygon_gdf.empty:
        return polygon_gdf

    odbl = odbl_gdf[odbl_gdf.geometry.notnull()].copy().reset_index(drop=True)
    polygon = polygon_gdf[polygon_gdf.geometry.notnull()].copy().reset_index(drop=True)
    if odbl.empty or polygon.empty:
        return polygon

    spatial_index = odbl.sindex
    keep_mask: list[bool] = []

    for geometry in polygon.geometry:
        if geometry is None:
            keep_mask.append(False)
            continue

        representative = geometry.representative_point()
        drop_feature = False
        candidate_ids = list(spatial_index.intersection(geometry.bounds))
        for candidate_id in candidate_ids:
            reference = odbl.geometry.iloc[candidate_id]
            if reference is None:
                continue
            if reference.contains(representative):
                drop_feature = True
                break

            intersection = clean_geometry(geometry.intersection(reference))
            if intersection is None:
                continue

            denominator = max(min(float(geometry.area), float(reference.area)), 1e-6)
            overlap_ratio = float(intersection.area) / denominator
            if overlap_ratio >= 0.55:
                drop_feature = True
                break

        keep_mask.append(not drop_feature)

    return polygon[pd.Series(keep_mask, index=polygon.index)].copy()


def smooth_building_geometry(geometry: BaseGeometry, source: Any) -> BaseGeometry | None:
    geometry = clean_geometry(geometry)
    if geometry is None:
        return None

    area = max(float(geometry.area), 1.0)
    span = area ** 0.5
    source_name = normalize_tag_value(source)
    is_machine_polygon = source_name not in {"osm", "ms", "microsoft"}

    simplify_tolerance = min(
        max(span * (0.018 if is_machine_polygon else 0.006), 0.16 if is_machine_polygon else 0.06),
        0.75 if is_machine_polygon else 0.22,
    )
    smoothed = clean_geometry(geometry.simplify(simplify_tolerance, preserve_topology=True))
    if smoothed is None:
        smoothed = geometry

    if area >= 16.0:
        smooth_buffer = min(
            max(span * (0.009 if is_machine_polygon else 0.003), 0.12 if is_machine_polygon else 0.04),
            0.35 if is_machine_polygon else 0.1,
        )
        rounded = clean_geometry(
            smoothed.buffer(smooth_buffer, join_style=1).buffer(-smooth_buffer * 0.94, join_style=1)
        )
        if rounded is not None:
            smoothed = rounded

    return clean_geometry(smoothed)


def enrich_buildings(
    odbl_path: Path,
    polygon_path: Path,
    lod1_json_path: Path,
) -> gpd.GeoDataFrame:
    odbl_gdf = gpd.read_file(odbl_path)
    polygon_gdf = gpd.read_file(polygon_path)

    if odbl_gdf.empty and polygon_gdf.empty:
        raise SystemExit("Indirilen tile icinde bina verisi bulunamadi.")

    base_crs = odbl_gdf.crs or polygon_gdf.crs or "EPSG:3857"
    odbl_gdf = odbl_gdf.to_crs(base_crs) if not odbl_gdf.empty else odbl_gdf
    polygon_gdf = polygon_gdf.to_crs(base_crs) if not polygon_gdf.empty else polygon_gdf
    polygon_gdf = drop_duplicate_polygon_buildings(odbl_gdf, polygon_gdf)

    frames: list[gpd.GeoDataFrame] = []
    if not odbl_gdf.empty:
        frames.append(odbl_gdf)
    if not polygon_gdf.empty:
        frames.append(polygon_gdf)

    buildings = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=base_crs)
    buildings = buildings[buildings.geometry.notnull()].copy()
    buildings["geometry"] = buildings.apply(
        lambda row: smooth_building_geometry(row.geometry, row.get("source")),
        axis=1,
    )
    buildings = buildings[buildings.geometry.notnull()].copy()

    lookup = load_height_lookup(lod1_json_path)

    def enrich_row(row: pd.Series) -> tuple[float | None, float | None]:
        key = f"{row.get('source', '')}{row.get('id', '')}{row.get('region', '')}"
        item = lookup.get(key, {})
        height = item.get("height")
        variance = item.get("var")
        return (
            float(height) if height is not None else None,
            float(variance) if variance is not None else None,
        )

    enriched = buildings.apply(enrich_row, axis=1, result_type="expand")
    buildings["height"] = enriched[0]
    buildings["var"] = enriched[1]
    return buildings


def pick_sample_center(
    buildings_gdf: gpd.GeoDataFrame,
    window_size_m: float,
    sample_center_lat: float | None,
    sample_center_lon: float | None,
) -> tuple[float, float]:
    if sample_center_lat is not None and sample_center_lon is not None:
        transformer = Transformer.from_crs("EPSG:4326", buildings_gdf.crs, always_xy=True)
        x, y = transformer.transform(sample_center_lon, sample_center_lat)
        return float(x), float(y)

    half_size = window_size_m / 2.0
    centroids = buildings_gdf.geometry.centroid
    coords = [(float(point.x), float(point.y)) for point in centroids]

    best_count = -1
    best_center = coords[0]
    for center_x, center_y in coords:
        count = sum(
            1
            for x, y in coords
            if abs(x - center_x) <= half_size and abs(y - center_y) <= half_size
        )
        if count > best_count:
            best_count = count
            best_center = (center_x, center_y)

    return best_center


def extract_sample(
    buildings_gdf: gpd.GeoDataFrame,
    center_x: float,
    center_y: float,
    window_size_m: float,
) -> tuple[gpd.GeoDataFrame, BaseGeometry]:
    half_size = window_size_m / 2.0
    sample_window = box(center_x - half_size, center_y - half_size, center_x + half_size, center_y + half_size)
    sample = buildings_gdf[buildings_gdf.intersects(sample_window)].copy()
    sample = sample[sample.geometry.notnull()].copy()
    if sample.empty:
        raise SystemExit("Secilen ornek penceresinde bina bulunamadi.")
    return sample, sample_window


def export_geojson(sample_gdf: gpd.GeoDataFrame, destination: Path) -> Path:
    ensure_directory(destination.parent)
    sample_gdf.to_crs("EPSG:4326").to_file(destination, driver="GeoJSON")
    return destination


def build_building_preview_mesh(
    sample_gdf: gpd.GeoDataFrame,
    target_size_mm: float,
    base_thickness_mm: float,
    height_exaggeration: float,
    min_building_height_mm: float,
) -> trimesh.Trimesh:
    minx, miny, maxx, maxy = sample_gdf.total_bounds
    width_m = max(float(maxx - minx), 1.0)
    depth_m = max(float(maxy - miny), 1.0)
    model_scale = target_size_mm / max(width_m, depth_m)

    shifted = sample_gdf.copy()
    shifted["geometry"] = shifted.geometry.apply(lambda geom: translate(geom, xoff=-minx, yoff=-miny))
    shifted["geometry"] = shifted.geometry.apply(
        lambda geom: scale_geometry(geom, xfact=model_scale, yfact=model_scale, origin=(0.0, 0.0)),
    )

    meshes: list[trimesh.Trimesh] = []
    margin_mm = 2.0
    plate = trimesh.creation.box(
        extents=(width_m * model_scale + margin_mm * 2.0, depth_m * model_scale + margin_mm * 2.0, base_thickness_mm)
    )
    plate.apply_translation(
        [
            width_m * model_scale / 2.0,
            depth_m * model_scale / 2.0,
            base_thickness_mm / 2.0,
        ]
    )
    meshes.append(plate)

    for _, row in shifted.iterrows():
        geometry = clean_geometry(row.geometry)
        if geometry is None:
            continue

        height_m = row.get("height")
        if height_m is None or pd.isna(height_m):
            continue

        height_mm = max(float(height_m) * model_scale * height_exaggeration, float(min_building_height_mm))
        polygons = [geometry] if geometry.geom_type == "Polygon" else list(getattr(geometry, "geoms", []))
        for polygon in polygons:
            polygon = clean_geometry(polygon)
            if polygon is None or polygon.area <= 0:
                continue
            mesh = trimesh.creation.extrude_polygon(polygon, height_mm)
            mesh.apply_translation([0.0, 0.0, base_thickness_mm])
            meshes.append(mesh)

    if len(meshes) == 1:
        raise SystemExit("STL olusturmak icin extrude edilebilir bina bulunamadi.")

    return repair_mesh(trimesh.util.concatenate(meshes))


def boundary_polygon_in_crs(boundary_polygon_latlon: BaseGeometry, target_crs: Any) -> BaseGeometry:
    return gpd.GeoSeries([boundary_polygon_latlon], crs="EPSG:4326").to_crs(target_crs).iloc[0]


def build_gba_building_meshes(
    buildings_raw: gpd.GeoDataFrame,
    boundary_polygon_latlon: BaseGeometry,
    boundary_projected: BaseGeometry,
    terrain_context,
    min_area_m2: float,
    min_width_m: float,
    min_building_height_m: float,
    height_exaggeration: float,
    print_tuning,
    terrain_embed_m: float,
) -> tuple[gpd.GeoDataFrame, list[trimesh.Trimesh]]:
    if buildings_raw.empty:
        return buildings_raw, []

    source_boundary = boundary_polygon_in_crs(boundary_polygon_latlon, buildings_raw.crs or "EPSG:3857")
    buildings = buildings_raw.copy()
    buildings["geometry"] = buildings.geometry.apply(
        lambda geom: clean_geometry(geom.intersection(source_boundary)) if geom is not None else None
    )
    buildings = buildings[buildings.geometry.notnull()].copy()
    if buildings.empty:
        return buildings, []

    buildings = buildings.to_crs(terrain_context.utm_crs)
    buildings = buildings[buildings.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    buildings["geometry"] = buildings.geometry.apply(
        lambda geom: clean_geometry(geom.intersection(boundary_projected)) if geom is not None else None
    )
    buildings = buildings[buildings.geometry.notnull()].copy()
    if buildings.empty:
        return buildings, []

    buildings["footprint_area"] = buildings.geometry.area
    buildings = buildings[buildings["footprint_area"] >= min_area_m2].copy()
    if buildings.empty:
        return buildings, []

    buffer_hint = max(min_width_m * 0.45, 0.18)
    buildings["geometry"] = buildings.apply(
        lambda row: thicken_small_geometry(row.geometry, buffer_hint)
        if row["footprint_area"] < max(min_area_m2 * 3.0, 18.0)
        else row.geometry,
        axis=1,
    )
    buildings["footprint_area"] = buildings.geometry.area

    def resolve_height(row: pd.Series) -> float:
        raw_height = row.get("height")
        if raw_height is None or pd.isna(raw_height):
            return float(min_building_height_m)
        return max(float(raw_height) * height_exaggeration, float(min_building_height_m))

    buildings["height_m"] = buildings.apply(resolve_height, axis=1)
    buildings = translate_geodataframe(buildings, -terrain_context.x_origin, -terrain_context.y_origin)

    building_heights = buildings["height_m"] + terrain_embed_m
    meshes = create_meshes_for_geometries(
        buildings.geometry,
        building_heights,
        elevation_sampler=lambda geom, mode: sample_geometry_elevation(terrain_context, geom, mode) - terrain_embed_m,
        base_mode="min",
        min_area_m2=min_area_m2,
    )

    detail_meshes: list[trimesh.Trimesh] = []
    for _, row in buildings.iterrows():
        detail_meshes.extend(
            create_building_detail_meshes(
                geometry=row.geometry,
                base_z=sample_geometry_elevation(terrain_context, row.geometry, "min") - terrain_embed_m,
                height_m=float(row["height_m"] + terrain_embed_m),
                building_type=normalize_tag_value(row.get("building")) or "yes",
                scale_factor=terrain_context.scale_factor,
                print_tuning=print_tuning,
                min_area_m2=min_area_m2,
            )
        )

    return buildings, meshes + detail_meshes


def build_tree_feature_meshes(
    raw_points_gdf: gpd.GeoDataFrame,
    raw_rows_gdf: gpd.GeoDataFrame,
    boundary_polygon_latlon: BaseGeometry,
    terrain_context,
    print_tuning,
) -> tuple[int, list[trimesh.Trimesh]]:
    source_crs = raw_points_gdf.crs or raw_rows_gdf.crs or "EPSG:4326"
    boundary_source = boundary_polygon_in_crs(boundary_polygon_latlon, source_crs)
    meshes: list[trimesh.Trimesh] = []
    feature_count = 0

    scale_factor = max(float(terrain_context.scale_factor), 1e-6)
    spacing_m = max(print_tuning.tree_open_spacing_mm / scale_factor * 0.95, 2.4)
    canopy_radius_m = max(print_tuning.tree_open_canopy_diam_mm / 2.0 / scale_factor, 0.55)
    canopy_height_m = max(print_tuning.tree_open_canopy_height_mm / scale_factor, 1.0)
    trunk_radius_m = max(print_tuning.tree_trunk_diam_mm / 2.0 / scale_factor, canopy_radius_m * 0.22, 0.18)
    trunk_height_m = max(canopy_height_m * 0.48, 0.85)

    if raw_points_gdf is not None and not raw_points_gdf.empty:
        points = raw_points_gdf[raw_points_gdf.geometry.notnull()].copy()
        points["geometry"] = points.geometry.apply(
            lambda geom: clean_geometry(geom.intersection(boundary_source)) if geom is not None else None
        )
        points = points[points.geometry.notnull()].copy()
        if not points.empty:
            points = points.to_crs(terrain_context.utm_crs)
            points = translate_geodataframe(points, -terrain_context.x_origin, -terrain_context.y_origin)
            for _, row in points.head(160).iterrows():
                point = row.geometry
                if point is None or point.geom_type != "Point":
                    continue
                base_z = sample_geometry_elevation(terrain_context, point, "max")
                meshes.append(
                    create_tree_mesh(
                        x=float(point.x),
                        y=float(point.y),
                        base_z=base_z,
                        trunk_radius_m=trunk_radius_m,
                        trunk_height_m=trunk_height_m,
                        canopy_radius_m=canopy_radius_m,
                        canopy_height_m=canopy_height_m,
                        canopy_style="sphere",
                    )
                )
                feature_count += 1

    if raw_rows_gdf is not None and not raw_rows_gdf.empty and feature_count < 220:
        rows = raw_rows_gdf[raw_rows_gdf.geometry.notnull()].copy()
        rows["geometry"] = rows.geometry.apply(
            lambda geom: clean_geometry(geom.intersection(boundary_source)) if geom is not None else None
        )
        rows = rows[rows.geometry.notnull()].copy()
        if not rows.empty:
            rows = rows.to_crs(terrain_context.utm_crs)
            rows = translate_geodataframe(rows, -terrain_context.x_origin, -terrain_context.y_origin)
            for _, row in rows.iterrows():
                for line in iter_lines(row.geometry):
                    if line.length <= 0:
                        continue
                    distances = np.arange(0.0, float(line.length) + spacing_m, spacing_m)
                    for distance in distances:
                        if feature_count >= 220:
                            break
                        point = line.interpolate(float(min(distance, line.length)))
                        base_z = sample_geometry_elevation(terrain_context, point, "max")
                        meshes.append(
                            create_tree_mesh(
                                x=float(point.x),
                                y=float(point.y),
                                base_z=base_z,
                                trunk_radius_m=trunk_radius_m * 0.92,
                                trunk_height_m=trunk_height_m,
                                canopy_radius_m=canopy_radius_m * 0.95,
                                canopy_height_m=canopy_height_m * 0.95,
                                canopy_style="cone",
                            )
                        )
                        feature_count += 1
                    if feature_count >= 220:
                        break
                if feature_count >= 220:
                    break

    return feature_count, meshes


def build_showcase_road_layer_meshes(
    raw_gdf: gpd.GeoDataFrame,
    boundary_polygon_latlon: BaseGeometry,
    boundary_projected: BaseGeometry,
    terrain_context,
    building_exclusion_geometry: BaseGeometry | None,
    terrain_embed_m: float,
    min_area_m2: float,
    min_width_m: float,
    road_base_height_m: float,
    road_top_height_m: float,
    road_crown_inset_m: float,
) -> tuple[gpd.GeoDataFrame, list[trimesh.Trimesh]]:
    if raw_gdf is None or raw_gdf.empty:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=terrain_context.utm_crs), []

    source_boundary = boundary_polygon_in_crs(boundary_polygon_latlon, raw_gdf.crs or "EPSG:4326")
    gdf = raw_gdf[raw_gdf.geometry.notnull()].copy()
    gdf["geometry"] = gdf.geometry.apply(
        lambda geom: clean_geometry(geom.intersection(source_boundary)) if geom is not None else None
    )
    gdf = gdf[gdf.geometry.notnull()].copy()
    if gdf.empty:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=terrain_context.utm_crs), []

    gdf = gdf.to_crs(terrain_context.utm_crs)
    gdf = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString", "Polygon", "MultiPolygon"])].copy()
    if gdf.empty:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=terrain_context.utm_crs), []

    grouped_surfaces: dict[float, list[BaseGeometry]] = {}
    road_smoothing_m = max(min_width_m * 0.22, 0.2)
    building_clearance_m = max(min_width_m * 0.08, 0.22)

    for _, row in gdf.iterrows():
        visible_layer = estimate_visible_road_layer(row)
        if visible_layer is None:
            continue

        width_m = max(estimate_road_surface_width(row, fallback_width_m=min_width_m), min_width_m)
        width_m *= 1.08

        if normalize_tag_value(row.get("oneway")) == "yes" and width_m < min_width_m * 1.8:
            width_m *= 1.08

        if width_m >= min_width_m * 1.8:
            surface = build_dual_carriageway_geometry(
                row.geometry,
                width_m,
                clip_polygon=boundary_projected,
            )
        else:
            surface = geometry_to_polygonal_feature(
                row.geometry,
                width_m,
                clip_polygon=boundary_projected,
            )

        surface = clean_geometry(surface)
        if surface is None:
            continue

        rounded = clean_geometry(
            surface.buffer(road_smoothing_m, join_style=1).buffer(-road_smoothing_m * 0.96, join_style=1)
        )
        if rounded is not None:
            surface = rounded

        translated = translate(surface, xoff=-terrain_context.x_origin, yoff=-terrain_context.y_origin)
        translated = clean_geometry(translated)
        if translated is None:
            continue

        z_offset_m = round(float(visible_layer) * road_base_height_m * 1.5, 6)
        grouped_surfaces.setdefault(z_offset_m, []).append(translated)

    merged_records: list[dict[str, Any]] = []
    bridge_gap_m = max(road_smoothing_m * 1.8, 0.35)
    for z_offset_m, geometries in grouped_surfaces.items():
        merged_surface = clean_geometry(unary_union(geometries))
        if merged_surface is None:
            continue

        bridged = clean_geometry(
            merged_surface.buffer(bridge_gap_m, join_style=1).buffer(-bridge_gap_m * 0.985, join_style=1)
        )
        if bridged is not None:
            merged_surface = bridged

        for polygon in merge_polygon_geometries([merged_surface], min_area_m2=min_area_m2):
            polygon = clean_geometry(polygon)
            if polygon is None or polygon.area < min_area_m2:
                continue

            if building_exclusion_geometry is not None:
                carved = clean_geometry(
                    polygon.difference(building_exclusion_geometry.buffer(building_clearance_m, join_style=1))
                )
                if carved is not None:
                    polygon = carved

            polygon = clean_geometry(polygon)
            if polygon is None:
                continue

            smoothed_after_cut = clean_geometry(
                polygon.buffer(road_smoothing_m * 0.7, join_style=1).buffer(-road_smoothing_m * 0.66, join_style=1)
            )
            if smoothed_after_cut is not None:
                polygon = smoothed_after_cut

            polygon = clean_geometry(polygon)
            if polygon is None or polygon.area < min_area_m2:
                continue

            for piece in merge_polygon_geometries([polygon], min_area_m2=min_area_m2):
                if piece is not None and piece.area >= min_area_m2:
                    merged_records.append({"geometry": piece, "z_offset_m": z_offset_m})

    if not merged_records:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=terrain_context.utm_crs), []

    if len(merged_records) > 1:
        min_kept_area_m2 = max(min_area_m2 * 6.0, 120.0)
        merged_records = [
            record
            for record in merged_records
            if float(record["geometry"].area) >= min_kept_area_m2
        ]
        if not merged_records:
            return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=terrain_context.utm_crs), []

    roads_gdf = gpd.GeoDataFrame(merged_records, geometry="geometry", crs=terrain_context.utm_crs)
    meshes: list[trimesh.Trimesh] = []

    for _, row in roads_gdf.iterrows():
        z_offset_m = float(row["z_offset_m"])
        meshes.extend(
            create_surface_following_prism_meshes(
                row.geometry,
                bottom_surface_z_resolver=lambda x, y, z_offset_m=z_offset_m: (
                    sample_elevation(terrain_context, x, y) + z_offset_m - terrain_embed_m
                ),
                top_surface_z_resolver=lambda x, y, z_offset_m=z_offset_m: (
                    sample_elevation(terrain_context, x, y) + z_offset_m + road_base_height_m
                ),
                min_area_m2=min_area_m2,
            )
        )

        crown = clean_geometry(row.geometry.buffer(-road_crown_inset_m, join_style=1))
        if crown is not None and crown.area > row.geometry.area * 0.18:
            meshes.extend(
                create_surface_following_prism_meshes(
                    crown,
                    bottom_surface_z_resolver=lambda x, y, z_offset_m=z_offset_m: (
                        sample_elevation(terrain_context, x, y)
                        + z_offset_m
                        - terrain_embed_m
                        + road_base_height_m * 0.42
                    ),
                    top_surface_z_resolver=lambda x, y, z_offset_m=z_offset_m: (
                        sample_elevation(terrain_context, x, y)
                        + z_offset_m
                        - terrain_embed_m
                        + road_base_height_m * 0.42
                        + road_top_height_m
                    ),
                    min_area_m2=min_area_m2 * 0.4,
                )
            )

    return roads_gdf, meshes


def resolve_scene_output_path(args: argparse.Namespace, stl_dir: Path, slug: str) -> Path:
    if args.scene_output:
        output_path = Path(args.scene_output)
        ensure_directory(output_path.parent)
        return output_path
    return stl_dir / f"{slug}_scene.stl"


def build_rich_scene(
    sample_buildings_gdf: gpd.GeoDataFrame,
    boundary_polygon_latlon: BaseGeometry,
    args: argparse.Namespace,
    feature_cache_dir: Path,
) -> tuple[trimesh.Trimesh, dict[str, int]]:
    print_tuning = PRINT_PROFILES[args.print_profile]
    boundary_gdf = gpd.GeoSeries([boundary_polygon_latlon], crs="EPSG:4326")
    utm_crs = boundary_gdf.estimate_utm_crs()
    boundary_projected = boundary_gdf.to_crs(utm_crs).iloc[0]

    minx, miny, maxx, maxy = boundary_projected.bounds
    width_m = maxx - minx
    depth_m = maxy - miny
    scale_factor = args.target_size_mm / max(width_m, depth_m)

    min_feature_height_mm = (
        args.min_building_height_mm
        if args.min_building_height_mm is not None
        else print_tuning.min_building_height_mm
    )
    min_area_m2 = print_tuning.min_feature_area_mm2 / (scale_factor ** 2)
    min_width_m = print_tuning.min_feature_width_mm / scale_factor
    road_min_width_m = max(min_width_m * 1.45, 0.8 / scale_factor)
    min_building_height_m = min_feature_height_mm / scale_factor
    road_height_m = print_tuning.road_height_mm / scale_factor
    road_base_height_m = max(road_height_m * 2.35, 2.8 / scale_factor) * 0.5
    road_top_height_m = max(road_height_m * 1.75, 1.9 / scale_factor) * 0.5
    road_crown_inset_m = max(min_width_m * 0.18, 0.24 / scale_factor)
    water_height_m = print_tuning.water_height_mm / scale_factor
    park_height_m = print_tuning.park_height_mm / scale_factor
    terrain_embed_m = print_tuning.terrain_embed_mm / scale_factor * 1.18
    base_height_m = args.base_thickness_mm / scale_factor

    terrain_context = build_terrain_context(
        boundary_polygon_latlon=boundary_polygon_latlon,
        utm_crs=utm_crs,
        scale_factor=scale_factor,
        zoom=args.terrain_zoom,
        max_size=args.terrain_max_size,
        z_scale=args.terrain_z_scale,
        smoothing_sigma=args.terrain_smoothing,
    )
    terrain_mesh = build_terrain_mesh(terrain_context, base_height=base_height_m)

    layer_meshes: list[trimesh.Trimesh] = [terrain_mesh]
    layer_counts: dict[str, int] = {"terrain": 1}

    buildings_gdf, building_meshes = build_gba_building_meshes(
        buildings_raw=sample_buildings_gdf,
        boundary_polygon_latlon=boundary_polygon_latlon,
        boundary_projected=boundary_projected,
        terrain_context=terrain_context,
        min_area_m2=min_area_m2,
        min_width_m=min_width_m,
        min_building_height_m=min_building_height_m,
        height_exaggeration=args.height_exaggeration,
        print_tuning=print_tuning,
        terrain_embed_m=terrain_embed_m,
    )
    layer_meshes.extend(building_meshes)
    layer_counts["buildings"] = len(buildings_gdf)
    building_exclusion_geometry = (
        clean_geometry(unary_union(list(buildings_gdf.geometry)))
        if buildings_gdf is not None and not buildings_gdf.empty
        else None
    )

    if not args.without_roads:
        roads_raw = fetch_osm_layer("gba_roads", boundary_polygon_latlon, ROAD_TAGS, feature_cache_dir)
        roads_raw = roads_raw[roads_raw.get("highway").notnull()].copy() if not roads_raw.empty else roads_raw
        if not roads_raw.empty:
            roads_raw = roads_raw[
                roads_raw["highway"].apply(lambda value: normalize_tag_value(value) not in ROAD_SKIP_VALUES)
            ].copy()

        roads_gdf, road_meshes = build_showcase_road_layer_meshes(
            raw_gdf=roads_raw,
            boundary_polygon_latlon=boundary_polygon_latlon,
            boundary_projected=boundary_projected,
            terrain_context=terrain_context,
            building_exclusion_geometry=building_exclusion_geometry,
            terrain_embed_m=terrain_embed_m,
            min_area_m2=min_area_m2,
            min_width_m=road_min_width_m,
            road_base_height_m=road_base_height_m,
            road_top_height_m=road_top_height_m,
            road_crown_inset_m=road_crown_inset_m,
        )
        layer_meshes.extend(road_meshes)
        layer_counts["roads"] = len(roads_gdf)

    if not args.without_water:
        water_polygon_raw = fetch_osm_layer(
            "gba_water_polygons",
            boundary_polygon_latlon,
            WATER_POLYGON_TAGS,
            feature_cache_dir,
        )
        water_line_raw = fetch_osm_layer(
            "gba_water_lines",
            boundary_polygon_latlon,
            WATER_LINE_TAGS,
            feature_cache_dir,
        )
        water_polygons_gdf, water_polygon_meshes = build_polygon_layer_meshes(
            raw_gdf=water_polygon_raw,
            boundary_polygon=boundary_polygon_latlon,
            boundary_projected=boundary_projected,
            terrain_context=terrain_context,
            terrain_embed_m=terrain_embed_m,
            min_area_m2=min_area_m2,
            height_m=water_height_m,
        )
        water_lines_gdf, water_line_meshes = build_buffered_linear_layer_meshes(
            raw_gdf=water_line_raw,
            boundary_polygon=boundary_polygon_latlon,
            boundary_projected=boundary_projected,
            terrain_context=terrain_context,
            terrain_embed_m=terrain_embed_m,
            min_area_m2=min_area_m2,
            min_width_m=road_min_width_m,
            height_m=water_height_m,
            width_resolver=lambda row, fallback: max(
                parse_length_to_meters(row.get("width")) or water_width_from_waterway(row.get("waterway")),
                fallback,
            ),
        )
        layer_meshes.extend(water_polygon_meshes)
        layer_meshes.extend(water_line_meshes)
        layer_counts["water"] = len(water_polygons_gdf) + len(water_lines_gdf)

    if not args.without_parks:
        parks_raw = fetch_osm_layer("gba_parks", boundary_polygon_latlon, PARK_TAGS, feature_cache_dir)
        parks_gdf, park_meshes = build_park_layer_meshes(
            raw_gdf=parks_raw,
            boundary_polygon=boundary_polygon_latlon,
            boundary_projected=boundary_projected,
            terrain_context=terrain_context,
            print_tuning=print_tuning,
            terrain_embed_m=terrain_embed_m,
            min_area_m2=min_area_m2,
            height_m=park_height_m,
        )
        layer_meshes.extend(park_meshes)
        layer_counts["parks"] = len(parks_gdf)

    tree_points_raw = fetch_osm_layer("gba_tree_points", boundary_polygon_latlon, TREE_POINT_TAGS, feature_cache_dir)
    tree_rows_raw = fetch_osm_layer("gba_tree_rows", boundary_polygon_latlon, TREE_ROW_TAGS, feature_cache_dir)
    tree_count, tree_meshes = build_tree_feature_meshes(
        raw_points_gdf=tree_points_raw,
        raw_rows_gdf=tree_rows_raw,
        boundary_polygon_latlon=boundary_polygon_latlon,
        terrain_context=terrain_context,
        print_tuning=print_tuning,
    )
    if tree_meshes:
        layer_meshes.extend(tree_meshes)
    layer_counts["trees"] = tree_count

    scene = trimesh.util.concatenate(layer_meshes)
    scene.apply_scale(scale_factor)
    scene = finalize_scene_mesh(scene)
    scene.apply_translation(-scene.bounds[0])
    return scene, layer_counts


def sample_summary(
    sample_gdf: gpd.GeoDataFrame,
    sample_window: BaseGeometry,
    region: str,
    tile: str,
    window_size_m: float,
    print_profile: str,
    layer_counts: dict[str, int] | None = None,
    scene_bounds_mm: dict[str, float] | None = None,
) -> dict[str, Any]:
    transformer = Transformer.from_crs(sample_gdf.crs, "EPSG:4326", always_xy=True)
    center_x = (sample_window.bounds[0] + sample_window.bounds[2]) / 2.0
    center_y = (sample_window.bounds[1] + sample_window.bounds[3]) / 2.0
    center_lon, center_lat = transformer.transform(center_x, center_y)
    min_lon, min_lat = transformer.transform(sample_window.bounds[0], sample_window.bounds[1])
    max_lon, max_lat = transformer.transform(sample_window.bounds[2], sample_window.bounds[3])

    height_stats = sample_gdf["height"].dropna()
    summary = {
        "region": region,
        "tile": tile,
        "window_size_m": float(window_size_m),
        "print_profile": print_profile,
        "building_count": int(len(sample_gdf)),
        "center_lat": round(float(center_lat), 6),
        "center_lon": round(float(center_lon), 6),
        "bbox_latlon": {
            "south": round(float(min_lat), 6),
            "west": round(float(min_lon), 6),
            "north": round(float(max_lat), 6),
            "east": round(float(max_lon), 6),
        },
        "height_stats_m": {
            "min": round(float(height_stats.min()), 3) if not height_stats.empty else None,
            "mean": round(float(height_stats.mean()), 3) if not height_stats.empty else None,
            "max": round(float(height_stats.max()), 3) if not height_stats.empty else None,
        },
    }
    if layer_counts is not None:
        summary["layer_counts"] = layer_counts
    if scene_bounds_mm is not None:
        summary["scene_bounds_mm"] = scene_bounds_mm
    return summary


def write_summary(summary: dict[str, Any], destination: Path) -> Path:
    ensure_directory(destination.parent)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return destination


def export_stl(mesh: trimesh.Trimesh, destination: Path) -> Path:
    ensure_directory(destination.parent)
    mesh.export(destination)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_osmnx(ensure_directory("cache"))
    feature_cache_dir = ensure_directory(Path("cache") / "features")
    download_dir = ensure_directory(args.download_dir)
    sample_dir = ensure_directory(args.sample_dir)
    stl_dir = ensure_directory(args.stl_dir)
    slug = sanitize_filename(f"gba_{args.region}_{args.tile}_{int(args.window_size_m)}m_{args.print_profile}")

    urls = dataset_urls(args.region, args.tile)
    odbl_path = download_file(urls["odbl"], download_dir / f"{args.tile}_odbl.geojson")
    polygon_path = download_file(urls["polygon"], download_dir / f"{args.tile}_polygon.geojson")
    lod1_json_path = download_file(urls["lod1"], download_dir / f"{args.tile}_lod1.json")

    buildings = enrich_buildings(odbl_path, polygon_path, lod1_json_path)
    center_x, center_y = pick_sample_center(
        buildings,
        window_size_m=args.window_size_m,
        sample_center_lat=args.sample_center_lat,
        sample_center_lon=args.sample_center_lon,
    )
    sample, sample_window = extract_sample(buildings, center_x, center_y, args.window_size_m)
    boundary_polygon_latlon = gpd.GeoSeries([sample_window], crs=sample.crs).to_crs("EPSG:4326").iloc[0]

    geojson_path = export_geojson(sample, sample_dir / f"{slug}.geojson")

    preview_min_height_mm = args.min_building_height_mm or 1.0
    preview_mesh = build_building_preview_mesh(
        sample,
        target_size_mm=args.target_size_mm,
        base_thickness_mm=max(args.base_thickness_mm * 0.45, 2.4),
        height_exaggeration=args.height_exaggeration,
        min_building_height_mm=preview_min_height_mm,
    )
    preview_stl_path = export_stl(preview_mesh, stl_dir / f"{slug}_buildings.stl")

    rich_scene_mesh, layer_counts = build_rich_scene(
        sample_buildings_gdf=sample,
        boundary_polygon_latlon=boundary_polygon_latlon,
        args=args,
        feature_cache_dir=feature_cache_dir,
    )
    scene_stl_path = export_stl(rich_scene_mesh, resolve_scene_output_path(args, stl_dir, slug))

    bounds = rich_scene_mesh.bounds
    scene_bounds_mm = {
        "width": round(float(bounds[1][0] - bounds[0][0]), 2),
        "depth": round(float(bounds[1][1] - bounds[0][1]), 2),
        "height": round(float(bounds[1][2] - bounds[0][2]), 2),
    }
    summary = sample_summary(
        sample,
        sample_window,
        args.region,
        args.tile,
        window_size_m=args.window_size_m,
        print_profile=args.print_profile,
        layer_counts=layer_counts,
        scene_bounds_mm=scene_bounds_mm,
    )
    summary_path = write_summary(summary, sample_dir / f"{slug}_summary.json")

    print(f"Ornek GeoJSON: {geojson_path}")
    print(f"Bina onizleme STL: {preview_stl_path}")
    print(f"Detayli sahne STL: {scene_stl_path}")
    print(f"Ozet JSON: {summary_path}")
    print(f"Bina sayisi: {summary['building_count']}")
    print(f"Merkez: {summary['center_lat']}, {summary['center_lon']}")
    print(f"BBox: {summary['bbox_latlon']}")
    print(f"Katmanlar: {layer_counts}")
    print(f"Sahne boyutlari (mm): {scene_bounds_mm}")
    print(f"Yukseklik istatistikleri (m): {summary['height_stats_m']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
