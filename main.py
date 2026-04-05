from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import osmnx as ox
import trimesh
from shapely.affinity import translate
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from terrain import build_terrain_context, build_terrain_mesh, sample_elevation, sample_geometry_elevation
from utils import (
    build_dual_carriageway_geometry,
    build_bbox_polygon,
    choose_buildings,
    clean_geometry,
    clip_geodataframe,
    create_meshes_for_geometries,
    create_surface_following_prism_meshes,
    create_tree_mesh,
    empty_gdf,
    estimate_railway_surface_width,
    estimate_road_surface_width,
    estimate_visible_transport_layer,
    estimate_visible_road_layer,
    ensure_directory,
    extrude_geometry_at_base,
    geometry_dimensions,
    geometry_to_polygonal_feature,
    get_height,
    get_place_boundary_polygon,
    get_place_center,
    load_gdf_cache,
    make_cache_key,
    make_circle_polygon,
    normalize_tag_value,
    parse_coordinate_polygon,
    parse_length_to_meters,
    project_geodataframe,
    repair_mesh,
    sanitize_filename,
    sample_points_in_polygon_grid,
    save_gdf_cache,
    should_split_dual_carriageway,
    thicken_small_geometry,
    translate_geodataframe,
    merge_polygon_geometries,
    water_width_from_waterway,
)


BUILDING_TAGS = {"building": True}
ROAD_TAGS = {"highway": True}
RAILWAY_TAGS = {"railway": True}
WATER_POLYGON_TAGS = {
    "natural": ["water"],
    "water": True,
    "landuse": ["reservoir", "basin"],
}
WATER_LINE_TAGS = {"waterway": True}
PARK_TAGS = {
    "leisure": ["park", "garden", "playground", "nature_reserve"],
    "landuse": ["grass", "recreation_ground", "village_green", "meadow"],
    "natural": ["wood", "scrub"],
}

ROAD_SKIP_VALUES = {
    "bus_guideway",
    "construction",
    "corridor",
    "elevator",
    "escalator",
    "planned",
    "platform",
    "proposed",
    "raceway",
}

RAILWAY_SKIP_VALUES = {
    "abandoned",
    "construction",
    "dismantled",
    "disused",
    "platform",
    "proposed",
    "razed",
}


@dataclass(frozen=True)
class PrintTuning:
    profile_name: str
    min_building_height_mm: float
    road_height_mm: float
    railway_height_mm: float
    water_height_mm: float
    park_height_mm: float
    terrain_embed_mm: float
    min_feature_area_mm2: float
    min_feature_width_mm: float
    building_detail_min_span_mm: float
    building_detail_outset_mm: float
    building_detail_height_mm: float
    parapet_inset_mm: float
    roof_crown_inset_mm: float
    tree_dense_spacing_mm: float
    tree_open_spacing_mm: float
    tree_dense_canopy_diam_mm: float
    tree_open_canopy_diam_mm: float
    tree_dense_canopy_height_mm: float
    tree_open_canopy_height_mm: float
    tree_trunk_diam_mm: float


PRINT_PROFILES: dict[str, PrintTuning] = {
    "balanced": PrintTuning(
        profile_name="balanced",
        min_building_height_mm=2.2,
        road_height_mm=0.7,
        railway_height_mm=0.9,
        water_height_mm=0.5,
        park_height_mm=0.45,
        terrain_embed_mm=0.22,
        min_feature_area_mm2=0.7,
        min_feature_width_mm=0.45,
        building_detail_min_span_mm=1.8,
        building_detail_outset_mm=0.45,
        building_detail_height_mm=0.55,
        parapet_inset_mm=0.3,
        roof_crown_inset_mm=0.8,
        tree_dense_spacing_mm=2.0,
        tree_open_spacing_mm=2.8,
        tree_dense_canopy_diam_mm=1.4,
        tree_open_canopy_diam_mm=1.7,
        tree_dense_canopy_height_mm=1.1,
        tree_open_canopy_height_mm=1.35,
        tree_trunk_diam_mm=0.3,
    ),
    "fdm": PrintTuning(
        profile_name="fdm",
        min_building_height_mm=3.2,
        road_height_mm=1.0,
        railway_height_mm=1.2,
        water_height_mm=0.9,
        park_height_mm=0.8,
        terrain_embed_mm=0.38,
        min_feature_area_mm2=1.8,
        min_feature_width_mm=0.9,
        building_detail_min_span_mm=4.0,
        building_detail_outset_mm=0.9,
        building_detail_height_mm=1.0,
        parapet_inset_mm=0.7,
        roof_crown_inset_mm=1.6,
        tree_dense_spacing_mm=4.0,
        tree_open_spacing_mm=5.0,
        tree_dense_canopy_diam_mm=3.2,
        tree_open_canopy_diam_mm=4.0,
        tree_dense_canopy_height_mm=2.6,
        tree_open_canopy_height_mm=3.0,
        tree_trunk_diam_mm=1.1,
    ),
    "resin": PrintTuning(
        profile_name="resin",
        min_building_height_mm=1.4,
        road_height_mm=0.5,
        railway_height_mm=0.65,
        water_height_mm=0.4,
        park_height_mm=0.35,
        terrain_embed_mm=0.14,
        min_feature_area_mm2=0.35,
        min_feature_width_mm=0.25,
        building_detail_min_span_mm=1.1,
        building_detail_outset_mm=0.28,
        building_detail_height_mm=0.38,
        parapet_inset_mm=0.2,
        roof_crown_inset_mm=0.55,
        tree_dense_spacing_mm=1.4,
        tree_open_spacing_mm=1.9,
        tree_dense_canopy_diam_mm=1.0,
        tree_open_canopy_diam_mm=1.25,
        tree_dense_canopy_height_mm=0.85,
        tree_open_canopy_height_mm=1.0,
        tree_trunk_diam_mm=0.45,
    ),
    "preview": PrintTuning(
        profile_name="preview",
        min_building_height_mm=0.9,
        road_height_mm=0.18,
        railway_height_mm=0.24,
        water_height_mm=0.12,
        park_height_mm=0.12,
        terrain_embed_mm=0.08,
        min_feature_area_mm2=0.05,
        min_feature_width_mm=0.18,
        building_detail_min_span_mm=0.8,
        building_detail_outset_mm=0.12,
        building_detail_height_mm=0.16,
        parapet_inset_mm=0.12,
        roof_crown_inset_mm=0.28,
        tree_dense_spacing_mm=0.9,
        tree_open_spacing_mm=1.35,
        tree_dense_canopy_diam_mm=0.28,
        tree_open_canopy_diam_mm=0.42,
        tree_dense_canopy_height_mm=0.24,
        tree_open_canopy_height_mm=0.34,
        tree_trunk_diam_mm=0.08,
    ),
}


def configure_osmnx(cache_root: Path) -> None:
    http_cache = ensure_directory(cache_root / "http")
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(http_cache)
    ox.settings.log_console = False
    ox.settings.requests_timeout = 120
    ox.settings.overpass_rate_limit = True


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--print-profile", choices=sorted(PRINT_PROFILES), default="balanced")
    common.add_argument("--selection", choices=["all", "largest", "random"], default="all")
    common.add_argument("--max-buildings", type=int, default=None)
    common.add_argument("--target-size-mm", type=float, default=160.0)
    common.add_argument("--terrain-zoom", type=int, default=13)
    common.add_argument("--terrain-max-size", type=int, default=260)
    common.add_argument("--terrain-z-scale", type=float, default=1.7)
    common.add_argument("--terrain-smoothing", type=float, default=0.9)
    common.add_argument("--base-thickness-mm", type=float, default=6.0)
    common.add_argument("--building-height-scale", type=float, default=1.0)
    common.add_argument("--min-building-height-mm", type=float, default=None)
    common.add_argument("--road-height-mm", type=float, default=None)
    common.add_argument("--railway-height-mm", type=float, default=None)
    common.add_argument("--water-height-mm", type=float, default=None)
    common.add_argument("--park-height-mm", type=float, default=None)
    common.add_argument("--terrain-embed-mm", type=float, default=None)
    common.add_argument("--min-feature-area-mm2", type=float, default=None)
    common.add_argument("--min-feature-width-mm", type=float, default=None)
    common.add_argument("--without-buildings", action="store_true")
    common.add_argument("--without-roads", action="store_true")
    common.add_argument("--without-railways", action="store_true")
    common.add_argument("--without-water", action="store_true")
    common.add_argument("--without-parks", action="store_true")
    common.add_argument("--output", default=None)
    common.add_argument("--output-dir", default="3_boyutlu_stl_dosyalarÄ±")

    parser = argparse.ArgumentParser(
        description="OSM bina, yol, demiryolu, su, park ve topografya katmanlarindan tek STL uretir.",
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    place_parser = subparsers.add_parser("place-radius", help="Yer adi + yariicap", parents=[common])
    place_parser.add_argument("place")
    place_parser.add_argument("radius", type=float)

    place_boundary_parser = subparsers.add_parser(
        "place-boundary",
        help="Yer adinin idari siniri",
        parents=[common],
    )
    place_boundary_parser.add_argument("place")

    point_parser = subparsers.add_parser("point-radius", help="Lat lon + yariicap", parents=[common])
    point_parser.add_argument("lat", type=float)
    point_parser.add_argument("lon", type=float)
    point_parser.add_argument("radius", type=float)

    bbox_parser = subparsers.add_parser("bbox", help="North south east west dikdortgen", parents=[common])
    bbox_parser.add_argument("north", type=float)
    bbox_parser.add_argument("south", type=float)
    bbox_parser.add_argument("east", type=float)
    bbox_parser.add_argument("west", type=float)

    polygon_parser = subparsers.add_parser("polygon", help="lat,lon; lat,lon; ... polygonu", parents=[common])
    polygon_parser.add_argument("coordinates")
    return parser


def resolve_print_tuning(args: argparse.Namespace) -> PrintTuning:
    base = PRINT_PROFILES[args.print_profile]
    return PrintTuning(
        profile_name=base.profile_name,
        min_building_height_mm=(
            args.min_building_height_mm if args.min_building_height_mm is not None else base.min_building_height_mm
        ),
        road_height_mm=args.road_height_mm if args.road_height_mm is not None else base.road_height_mm,
        railway_height_mm=args.railway_height_mm if args.railway_height_mm is not None else base.railway_height_mm,
        water_height_mm=args.water_height_mm if args.water_height_mm is not None else base.water_height_mm,
        park_height_mm=args.park_height_mm if args.park_height_mm is not None else base.park_height_mm,
        terrain_embed_mm=args.terrain_embed_mm if args.terrain_embed_mm is not None else base.terrain_embed_mm,
        min_feature_area_mm2=(
            args.min_feature_area_mm2 if args.min_feature_area_mm2 is not None else base.min_feature_area_mm2
        ),
        min_feature_width_mm=(
            args.min_feature_width_mm if args.min_feature_width_mm is not None else base.min_feature_width_mm
        ),
        building_detail_min_span_mm=base.building_detail_min_span_mm,
        building_detail_outset_mm=base.building_detail_outset_mm,
        building_detail_height_mm=base.building_detail_height_mm,
        parapet_inset_mm=base.parapet_inset_mm,
        roof_crown_inset_mm=base.roof_crown_inset_mm,
        tree_dense_spacing_mm=base.tree_dense_spacing_mm,
        tree_open_spacing_mm=base.tree_open_spacing_mm,
        tree_dense_canopy_diam_mm=base.tree_dense_canopy_diam_mm,
        tree_open_canopy_diam_mm=base.tree_open_canopy_diam_mm,
        tree_dense_canopy_height_mm=base.tree_dense_canopy_height_mm,
        tree_open_canopy_height_mm=base.tree_open_canopy_height_mm,
        tree_trunk_diam_mm=base.tree_trunk_diam_mm,
    )


def resolve_area(args: argparse.Namespace) -> tuple[BaseGeometry, str, dict[str, Any]]:
    if args.mode == "place-radius":
        lat, lon = get_place_center(args.place)
        polygon = make_circle_polygon(lat, lon, args.radius)
        label = sanitize_filename(f"place_radius_{args.place}_{int(args.radius)}m")
        return polygon, label, {"place": args.place, "center": (lat, lon), "radius_m": args.radius}

    if args.mode == "place-boundary":
        polygon = get_place_boundary_polygon(args.place)
        label = sanitize_filename(f"place_boundary_{args.place}")
        return polygon, label, {"place": args.place, "boundary_mode": "administrative"}

    if args.mode == "point-radius":
        polygon = make_circle_polygon(args.lat, args.lon, args.radius)
        label = sanitize_filename(
            f"point_radius_{round(args.lat, 5)}_{round(args.lon, 5)}_{int(args.radius)}m"
        )
        return polygon, label, {"center": (args.lat, args.lon), "radius_m": args.radius}

    if args.mode == "bbox":
        polygon = build_bbox_polygon(args.north, args.south, args.east, args.west)
        label = sanitize_filename(
            f"bbox_n{args.north}_s{args.south}_e{args.east}_w{args.west}".replace(".", "_")
        )
        return polygon, label, {
            "north": args.north,
            "south": args.south,
            "east": args.east,
            "west": args.west,
        }

    polygon = parse_coordinate_polygon(args.coordinates)
    label = sanitize_filename(f"polygon_{len(polygon.exterior.coords) - 1}_nokta")
    return polygon, label, {"points": len(polygon.exterior.coords) - 1}


def fetch_osm_layer(
    layer_name: str,
    polygon: BaseGeometry,
    tags: dict[str, bool | str | list[str]],
    cache_dir: Path,
) -> gpd.GeoDataFrame:
    payload = {
        "layer": layer_name,
        "tags": tags,
        "polygon_wkt": polygon.wkt,
    }
    cache_path = cache_dir / f"{layer_name}_{make_cache_key(payload)}.json"
    cached = load_gdf_cache(cache_path)
    if cached is not None:
        print(f"{layer_name}: cache'den yuklendi ({len(cached)} oge)")
        return cached

    try:
        gdf = ox.features_from_polygon(polygon, tags)
    except Exception as exc:
        if "No matching features" in str(exc):
            print(f"{layer_name}: eslesen veri bulunamadi")
            gdf = empty_gdf("EPSG:4326")
            save_gdf_cache(cache_path, gdf)
            return gdf
        raise SystemExit(f"{layer_name} verisi alinamadi: {exc}") from exc

    save_gdf_cache(cache_path, gdf)
    print(f"{layer_name}: indirildi ({len(gdf)} oge)")
    return gdf


def empty_projected_gdf(crs: Any) -> gpd.GeoDataFrame:
    return empty_gdf(crs)


def prepare_projected_features(
    raw_gdf: gpd.GeoDataFrame,
    boundary_polygon: BaseGeometry,
    utm_crs: Any,
) -> gpd.GeoDataFrame:
    if raw_gdf is None or raw_gdf.empty:
        return empty_projected_gdf(utm_crs)

    clipped = clip_geodataframe(raw_gdf, boundary_polygon)
    if clipped.empty:
        return empty_projected_gdf(utm_crs)

    projected = project_geodataframe(clipped, utm_crs)
    projected["geometry"] = projected.geometry.apply(clean_geometry)
    projected = projected[projected.geometry.notnull()].copy()
    return projected


def build_building_meshes(
    buildings_raw: gpd.GeoDataFrame,
    boundary_polygon: BaseGeometry,
    boundary_projected: BaseGeometry,
    terrain_context,
    args: argparse.Namespace,
    print_tuning: PrintTuning,
    terrain_embed_m: float,
    min_area_m2: float,
    min_width_m: float,
    min_building_height_m: float,
) -> tuple[gpd.GeoDataFrame, list[trimesh.Trimesh]]:
    buildings = prepare_projected_features(buildings_raw, boundary_polygon, terrain_context.utm_crs)
    if buildings.empty:
        return buildings, []

    buildings = buildings[buildings.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if buildings.empty:
        return buildings, []

    buildings["geometry"] = buildings.geometry.apply(
        lambda geom: clean_geometry(geom.intersection(boundary_projected))
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
    buildings = choose_buildings(buildings, args.selection, args.max_buildings)
    buildings["height_m"] = buildings.apply(
        lambda row: max(
            get_height(row, levels_multiplier=3.2, default_height=12.0) * args.building_height_scale,
            min_building_height_m,
        ),
        axis=1,
    )

    buildings = translate_geodataframe(
        buildings,
        x_offset=-terrain_context.x_origin,
        y_offset=-terrain_context.y_origin,
    )

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
                building_type=normalize_tag_value(row.get("building")),
                scale_factor=terrain_context.scale_factor,
                print_tuning=print_tuning,
                min_area_m2=min_area_m2,
            )
        )
    return buildings, meshes + detail_meshes


def create_building_detail_meshes(
    geometry: BaseGeometry,
    base_z: float,
    height_m: float,
    building_type: str | None,
    scale_factor: float,
    print_tuning: PrintTuning,
    min_area_m2: float,
) -> list[trimesh.Trimesh]:
    geometry = clean_geometry(geometry)
    if geometry is None:
        return []

    width, depth = geometry_dimensions(geometry)
    short_side = min(width, depth)
    area_m2 = geometry.area
    if area_m2 < max(min_area_m2 * 3.0, 40.0):
        return []
    if short_side < max(5.0, print_tuning.building_detail_min_span_mm / max(scale_factor, 1e-6)):
        return []

    meshes: list[trimesh.Trimesh] = []
    detail_outset_m = min(
        max(print_tuning.building_detail_outset_mm / max(scale_factor, 1e-6), 1.0),
        short_side * 0.08,
        4.5,
    )
    detail_height_m = min(
        max(print_tuning.building_detail_height_mm / max(scale_factor, 1e-6), 1.2),
        max(height_m * 0.14, 1.2),
    )
    parapet_inset_m = min(
        max(print_tuning.parapet_inset_mm / max(scale_factor, 1e-6), 0.8),
        short_side * 0.12,
    )
    roof_crown_inset_m = min(
        max(print_tuning.roof_crown_inset_mm / max(scale_factor, 1e-6), 1.5),
        short_side * 0.18,
    )

    inner_roof = clean_geometry(geometry.buffer(-parapet_inset_m, join_style=1))
    if inner_roof is not None:
        parapet_ring = clean_geometry(geometry.difference(inner_roof))
        if parapet_ring is not None:
            meshes.extend(
                extrude_geometry_at_base(
                    parapet_ring,
                    height_m=min(
                        detail_height_m * 0.8,
                        max(print_tuning.building_detail_height_mm / max(scale_factor, 1e-6), 0.8),
                    ),
                    base_z=base_z + height_m,
                    min_area_m2=min_area_m2 * 0.4,
                )
            )

    roof_crown = clean_geometry(geometry.buffer(-roof_crown_inset_m, join_style=1))
    if roof_crown is not None and roof_crown.area > area_m2 * 0.12:
        crown_height_m = min(max(height_m * 0.12, 0.9 / max(scale_factor, 1e-6)), max(height_m * 0.28, 1.2))
        meshes.extend(
            extrude_geometry_at_base(
                roof_crown,
                height_m=crown_height_m,
                base_z=base_z + height_m,
                min_area_m2=min_area_m2 * 0.5,
            )
        )

        second_crown = clean_geometry(roof_crown.buffer(-roof_crown_inset_m * 0.65, join_style=1))
        if second_crown is not None and second_crown.area > area_m2 * 0.05 and height_m > crown_height_m * 2.0:
            second_height_m = min(max(crown_height_m * 0.55, 0.6 / max(scale_factor, 1e-6)), crown_height_m)
            meshes.extend(
                extrude_geometry_at_base(
                    second_crown,
                    height_m=second_height_m,
                    base_z=base_z + height_m + crown_height_m * 0.55,
                    min_area_m2=min_area_m2 * 0.4,
                )
            )

    balcony_candidates = {"apartments", "residential", "house", "hotel", "dormitory", "yes"}
    if building_type in balcony_candidates or (height_m < 60.0 and area_m2 > 120.0):
        balcony_ring = clean_geometry(geometry.buffer(detail_outset_m, join_style=1).difference(geometry))
        if balcony_ring is not None and balcony_ring.area < area_m2 * 1.4:
            band_height_m = min(
                max(print_tuning.building_detail_height_mm / max(scale_factor, 1e-6), 1.0),
                detail_height_m,
            )
            level_ratios = [0.28, 0.54]
            if height_m > 28.0:
                level_ratios.append(0.78)
            for ratio in level_ratios:
                level_z = base_z + max(height_m * ratio, band_height_m)
                if level_z + band_height_m >= base_z + height_m * 0.98:
                    continue
                meshes.extend(
                    extrude_geometry_at_base(
                        balcony_ring,
                        height_m=band_height_m,
                        base_z=level_z,
                        min_area_m2=min_area_m2 * 0.35,
                    )
                )

    return meshes


def build_polygon_layer_meshes(
    raw_gdf: gpd.GeoDataFrame,
    boundary_polygon: BaseGeometry,
    boundary_projected: BaseGeometry,
    terrain_context,
    terrain_embed_m: float,
    min_area_m2: float,
    height_m: float,
) -> tuple[gpd.GeoDataFrame, list[trimesh.Trimesh]]:
    gdf = prepare_projected_features(raw_gdf, boundary_polygon, terrain_context.utm_crs)
    if gdf.empty:
        return gdf, []

    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if gdf.empty:
        return gdf, []

    gdf["geometry"] = gdf.geometry.apply(lambda geom: clean_geometry(geom.intersection(boundary_projected)))
    gdf = gdf[gdf.geometry.notnull()].copy()
    if gdf.empty:
        return gdf, []

    gdf["footprint_area"] = gdf.geometry.area
    gdf = gdf[gdf["footprint_area"] >= min_area_m2].copy()
    gdf = translate_geodataframe(gdf, -terrain_context.x_origin, -terrain_context.y_origin)

    meshes = create_meshes_for_geometries(
        gdf.geometry,
        [height_m + terrain_embed_m] * len(gdf),
        elevation_sampler=lambda geom, mode: sample_geometry_elevation(terrain_context, geom, mode) - terrain_embed_m,
        base_mode="min",
        min_area_m2=min_area_m2,
    )
    return gdf, meshes


def build_park_layer_meshes(
    raw_gdf: gpd.GeoDataFrame,
    boundary_polygon: BaseGeometry,
    boundary_projected: BaseGeometry,
    terrain_context,
    print_tuning: PrintTuning,
    terrain_embed_m: float,
    min_area_m2: float,
    height_m: float,
) -> tuple[gpd.GeoDataFrame, list[trimesh.Trimesh]]:
    parks = prepare_projected_features(raw_gdf, boundary_polygon, terrain_context.utm_crs)
    if parks.empty:
        return parks, []

    parks = parks[parks.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if parks.empty:
        return parks, []

    parks["geometry"] = parks.geometry.apply(lambda geom: clean_geometry(geom.intersection(boundary_projected)))
    parks = parks[parks.geometry.notnull()].copy()
    if parks.empty:
        return parks, []

    parks["footprint_area"] = parks.geometry.area
    parks = parks[parks["footprint_area"] >= min_area_m2].copy()
    if parks.empty:
        return parks, []

    parks = translate_geodataframe(parks, -terrain_context.x_origin, -terrain_context.y_origin)
    meshes = create_meshes_for_geometries(
        parks.geometry,
        [height_m + terrain_embed_m] * len(parks),
        elevation_sampler=lambda geom, mode: sample_geometry_elevation(terrain_context, geom, mode) - terrain_embed_m,
        base_mode="min",
        min_area_m2=min_area_m2,
    )

    tree_meshes: list[trimesh.Trimesh] = []
    scale_factor = max(terrain_context.scale_factor, 1e-6)
    for row_index, (_, row) in enumerate(parks.iterrows()):
        geometry = row.geometry
        area_m2 = geometry.area
        natural = normalize_tag_value(row.get("natural"))
        leisure = normalize_tag_value(row.get("leisure"))
        landuse = normalize_tag_value(row.get("landuse"))

        dense = natural in {"wood", "scrub"} or landuse in {"forest", "meadow"}
        spacing_m = (
            print_tuning.tree_dense_spacing_mm / scale_factor
            if dense
            else print_tuning.tree_open_spacing_mm / scale_factor
        )
        canopy_radius_m = (
            print_tuning.tree_dense_canopy_diam_mm / 2.0 / scale_factor
            if dense
            else print_tuning.tree_open_canopy_diam_mm / 2.0 / scale_factor
        )
        canopy_height_m = (
            print_tuning.tree_dense_canopy_height_mm / scale_factor
            if dense
            else print_tuning.tree_open_canopy_height_mm / scale_factor
        )
        trunk_radius_m = max(print_tuning.tree_trunk_diam_mm / 2.0 / scale_factor, canopy_radius_m * 0.18, 0.45)
        trunk_height_m = max(canopy_height_m * 0.42, 2.2)
        max_trees = min(220, max(1, int(area_m2 / max(spacing_m * spacing_m, 1.0))))
        margin_m = canopy_radius_m * 0.75
        points = sample_points_in_polygon_grid(
            geometry,
            spacing_m=spacing_m,
            margin_m=margin_m,
            max_points=max_trees,
        )

        for point_index, (x, y) in enumerate(points):
            canopy_style = "sphere" if (row_index + point_index) % 3 == 0 and leisure in {"park", "garden"} else "cone"
            point_geom = Point(x, y)
            point_base_z = sample_geometry_elevation(terrain_context, point_geom, "max") + height_m
            tree_meshes.append(
                create_tree_mesh(
                    x=x,
                    y=y,
                    base_z=point_base_z,
                    trunk_radius_m=trunk_radius_m,
                    trunk_height_m=trunk_height_m,
                    canopy_radius_m=canopy_radius_m,
                    canopy_height_m=canopy_height_m,
                    canopy_style=canopy_style,
                )
            )

    return parks, meshes + tree_meshes


def build_road_layer_meshes(
    raw_gdf: gpd.GeoDataFrame,
    boundary_polygon: BaseGeometry,
    boundary_projected: BaseGeometry,
    terrain_context,
    building_exclusion_geometry: BaseGeometry | None,
    terrain_embed_m: float,
    min_area_m2: float,
    min_width_m: float,
    height_m: float,
) -> tuple[gpd.GeoDataFrame, list[trimesh.Trimesh]]:
    gdf = prepare_projected_features(raw_gdf, boundary_polygon, terrain_context.utm_crs)
    if gdf.empty:
        return gdf, []

    gdf = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString", "Polygon", "MultiPolygon"])].copy()
    if gdf.empty:
        return gdf, []

    grouped_surfaces: dict[float, list[BaseGeometry]] = {}
    road_smoothing_m = max(min_width_m * 0.22, 0.18)
    bridge_gap_m = max(road_smoothing_m * 1.8, 0.35)
    building_clearance_m = max(min_width_m * 0.08, 0.2)

    for _, row in gdf.iterrows():
        visible_layer = estimate_visible_road_layer(row)
        if visible_layer is None:
            continue

        width_m = max(estimate_road_surface_width(row, fallback_width_m=min_width_m), min_width_m)
        if should_split_dual_carriageway(row, width_m):
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

        translated = translate(
            surface,
            xoff=-terrain_context.x_origin,
            yoff=-terrain_context.y_origin,
        )
        translated = clean_geometry(translated)
        if translated is None:
            continue

        z_offset_m = round(visible_layer * height_m * 1.5, 6)
        grouped_surfaces.setdefault(z_offset_m, []).append(translated)

    merged_records: list[dict[str, Any]] = []

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
            if polygon is None or polygon.area < min_area_m2:
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
                merged_records.append({"geometry": piece, "z_offset_m": z_offset_m})

    if not merged_records:
        return empty_projected_gdf(terrain_context.utm_crs), []

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
                    sample_elevation(terrain_context, x, y) + z_offset_m + height_m
                ),
                min_area_m2=min_area_m2,
            )
        )

    return roads_gdf, meshes


def build_railway_layer_meshes(
    raw_gdf: gpd.GeoDataFrame,
    boundary_polygon: BaseGeometry,
    boundary_projected: BaseGeometry,
    terrain_context,
    terrain_embed_m: float,
    min_area_m2: float,
    min_width_m: float,
    height_m: float,
) -> tuple[gpd.GeoDataFrame, list[trimesh.Trimesh]]:
    gdf = prepare_projected_features(raw_gdf, boundary_polygon, terrain_context.utm_crs)
    if gdf.empty:
        return gdf, []

    gdf = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString", "Polygon", "MultiPolygon"])].copy()
    if gdf.empty:
        return gdf, []

    grouped_surfaces: dict[float, list[BaseGeometry]] = {}
    smoothing_m = max(min_width_m * 0.2, 0.16)
    bridge_gap_m = max(smoothing_m * 1.55, 0.28)

    for _, row in gdf.iterrows():
        visible_layer = estimate_visible_transport_layer(row)
        if visible_layer is None:
            continue

        width_m = max(estimate_railway_surface_width(row, fallback_width_m=min_width_m), min_width_m)
        surface = geometry_to_polygonal_feature(
            row.geometry,
            width_m,
            clip_polygon=boundary_projected,
        )
        surface = clean_geometry(surface)
        if surface is None:
            continue

        rounded = clean_geometry(
            surface.buffer(smoothing_m, join_style=1).buffer(-smoothing_m * 0.95, join_style=1)
        )
        if rounded is not None:
            surface = rounded

        translated = translate(
            surface,
            xoff=-terrain_context.x_origin,
            yoff=-terrain_context.y_origin,
        )
        translated = clean_geometry(translated)
        if translated is None:
            continue

        z_offset_m = round(visible_layer * height_m * 1.35, 6)
        grouped_surfaces.setdefault(z_offset_m, []).append(translated)

    merged_records: list[dict[str, Any]] = []
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

            smoothed = clean_geometry(
                polygon.buffer(smoothing_m * 0.62, join_style=1).buffer(-smoothing_m * 0.58, join_style=1)
            )
            if smoothed is not None:
                polygon = smoothed

            polygon = clean_geometry(polygon)
            if polygon is None or polygon.area < min_area_m2:
                continue

            for piece in merge_polygon_geometries([polygon], min_area_m2=min_area_m2):
                merged_records.append({"geometry": piece, "z_offset_m": z_offset_m})

    if not merged_records:
        return empty_projected_gdf(terrain_context.utm_crs), []

    railways_gdf = gpd.GeoDataFrame(merged_records, geometry="geometry", crs=terrain_context.utm_crs)
    meshes: list[trimesh.Trimesh] = []

    for _, row in railways_gdf.iterrows():
        z_offset_m = float(row["z_offset_m"])
        meshes.extend(
            create_surface_following_prism_meshes(
                row.geometry,
                bottom_surface_z_resolver=lambda x, y, z_offset_m=z_offset_m: (
                    sample_elevation(terrain_context, x, y) + z_offset_m - terrain_embed_m
                ),
                top_surface_z_resolver=lambda x, y, z_offset_m=z_offset_m: (
                    sample_elevation(terrain_context, x, y) + z_offset_m + height_m
                ),
                min_area_m2=min_area_m2,
            )
        )

    return railways_gdf, meshes


def build_buffered_linear_layer_meshes(
    raw_gdf: gpd.GeoDataFrame,
    boundary_polygon: BaseGeometry,
    boundary_projected: BaseGeometry,
    terrain_context,
    terrain_embed_m: float,
    min_area_m2: float,
    min_width_m: float,
    height_m: float,
    width_resolver,
) -> tuple[gpd.GeoDataFrame, list[trimesh.Trimesh]]:
    gdf = prepare_projected_features(raw_gdf, boundary_polygon, terrain_context.utm_crs)
    if gdf.empty:
        return gdf, []

    gdf = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString", "Polygon", "MultiPolygon"])].copy()
    if gdf.empty:
        return gdf, []

    records: list[dict[str, Any]] = []
    for _, row in gdf.iterrows():
        width = width_resolver(row, min_width_m)
        polygonal = geometry_to_polygonal_feature(row.geometry, width, clip_polygon=boundary_projected)
        polygonal = clean_geometry(polygonal)
        if polygonal is None:
            continue

        translated = translate(
            polygonal,
            xoff=-terrain_context.x_origin,
            yoff=-terrain_context.y_origin,
        )
        translated = clean_geometry(translated)
        if translated is None:
            continue

        area = sum(poly.area for poly in _iter_polygons_quick(translated))
        if area < min_area_m2:
            continue

        records.append({"geometry": translated, "area_m2": area})

    if not records:
        return empty_projected_gdf(terrain_context.utm_crs), []

    prepared = gpd.GeoDataFrame(records, geometry="geometry", crs=terrain_context.utm_crs)
    meshes = create_meshes_for_geometries(
        prepared.geometry,
        [height_m + terrain_embed_m] * len(prepared),
        elevation_sampler=lambda geom, mode: sample_geometry_elevation(terrain_context, geom, mode) - terrain_embed_m,
        base_mode="min",
        min_area_m2=min_area_m2,
    )
    return prepared, meshes


def _iter_polygons_quick(geometry: BaseGeometry) -> Iterable[BaseGeometry]:
    if geometry.geom_type == "Polygon":
        yield geometry
        return
    if geometry.geom_type == "MultiPolygon":
        for polygon in geometry.geoms:
            yield polygon
        return
    if hasattr(geometry, "geoms"):
        for child in geometry.geoms:
            yield from _iter_polygons_quick(child)


def resolve_output_path(args: argparse.Namespace, label: str) -> Path:
    if args.output:
        output_path = Path(args.output)
        ensure_directory(output_path.parent)
        return output_path

    output_dir = ensure_directory(args.output_dir)
    return output_dir / f"scene_{args.mode}_{label}.stl"


def finalize_scene_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
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
    return mesh


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    print_tuning = resolve_print_tuning(args)

    cache_root = ensure_directory("cache")
    feature_cache_dir = ensure_directory(cache_root / "features")
    configure_osmnx(cache_root)

    boundary_polygon, label, area_info = resolve_area(args)
    boundary_gdf = gpd.GeoSeries([boundary_polygon], crs="EPSG:4326")
    utm_crs = boundary_gdf.estimate_utm_crs()
    boundary_projected = boundary_gdf.to_crs(utm_crs).iloc[0]

    minx, miny, maxx, maxy = boundary_projected.bounds
    width_m = maxx - minx
    depth_m = maxy - miny
    scale_factor = args.target_size_mm / max(width_m, depth_m)

    min_area_m2 = print_tuning.min_feature_area_mm2 / (scale_factor ** 2)
    min_width_m = print_tuning.min_feature_width_mm / scale_factor
    min_building_height_m = print_tuning.min_building_height_mm / scale_factor
    road_height_m = print_tuning.road_height_mm / scale_factor
    railway_height_m = print_tuning.railway_height_mm / scale_factor
    water_height_m = print_tuning.water_height_mm / scale_factor
    park_height_m = print_tuning.park_height_mm / scale_factor
    terrain_embed_m = print_tuning.terrain_embed_mm / scale_factor
    base_height_m = args.base_thickness_mm / scale_factor

    print("Alan ozeti:", area_info)
    print(
        "Model ayarlari:",
        {
            "print_profile": print_tuning.profile_name,
            "taban_genislik_m": round(width_m, 2),
            "taban_derinlik_m": round(depth_m, 2),
            "hedef_mm": args.target_size_mm,
            "olcek": round(scale_factor, 5),
            "min_alan_m2": round(min_area_m2, 3),
            "terrain_embed_mm": round(print_tuning.terrain_embed_mm, 3),
        },
    )

    terrain_context = build_terrain_context(
        boundary_polygon_latlon=boundary_polygon,
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

    buildings_gdf = empty_projected_gdf(utm_crs)

    if not args.without_buildings:
        buildings_raw = fetch_osm_layer("buildings", boundary_polygon, BUILDING_TAGS, feature_cache_dir)
        buildings_gdf, buildings_meshes = build_building_meshes(
            buildings_raw=buildings_raw,
            boundary_polygon=boundary_polygon,
            boundary_projected=boundary_projected,
            terrain_context=terrain_context,
            args=args,
            print_tuning=print_tuning,
            terrain_embed_m=terrain_embed_m,
            min_area_m2=min_area_m2,
            min_width_m=min_width_m,
            min_building_height_m=min_building_height_m,
        )
        layer_meshes.extend(buildings_meshes)
        layer_counts["buildings"] = len(buildings_gdf)
        print(f"Binalar: {len(buildings_gdf)} geometri, {len(buildings_meshes)} mesh")

    if not args.without_roads:
        roads_raw = fetch_osm_layer("roads", boundary_polygon, ROAD_TAGS, feature_cache_dir)
        roads_raw = roads_raw[roads_raw.get("highway").notnull()].copy() if not roads_raw.empty else roads_raw
        if not roads_raw.empty:
            roads_raw = roads_raw[
                roads_raw["highway"].apply(lambda value: normalize_tag_value(value) not in ROAD_SKIP_VALUES)
            ].copy()

        building_exclusion_geometry = (
            clean_geometry(unary_union(list(buildings_gdf.geometry))) if not buildings_gdf.empty else None
        )
        roads_gdf, road_meshes = build_road_layer_meshes(
            raw_gdf=roads_raw,
            boundary_polygon=boundary_polygon,
            boundary_projected=boundary_projected,
            terrain_context=terrain_context,
            building_exclusion_geometry=building_exclusion_geometry,
            terrain_embed_m=terrain_embed_m,
            min_area_m2=min_area_m2,
            min_width_m=min_width_m,
            height_m=road_height_m,
        )
        layer_meshes.extend(road_meshes)
        layer_counts["roads"] = len(roads_gdf)
        print(f"Yollar: {len(roads_gdf)} geometri, {len(road_meshes)} mesh")

    if not args.without_railways:
        railways_raw = fetch_osm_layer("railways", boundary_polygon, RAILWAY_TAGS, feature_cache_dir)
        railways_raw = railways_raw[railways_raw.get("railway").notnull()].copy() if not railways_raw.empty else railways_raw
        if not railways_raw.empty:
            railways_raw = railways_raw[
                railways_raw["railway"].apply(lambda value: normalize_tag_value(value) not in RAILWAY_SKIP_VALUES)
            ].copy()

        railways_gdf, railway_meshes = build_railway_layer_meshes(
            raw_gdf=railways_raw,
            boundary_polygon=boundary_polygon,
            boundary_projected=boundary_projected,
            terrain_context=terrain_context,
            terrain_embed_m=terrain_embed_m,
            min_area_m2=min_area_m2,
            min_width_m=min_width_m,
            height_m=railway_height_m,
        )
        layer_meshes.extend(railway_meshes)
        layer_counts["railways"] = len(railways_gdf)
        print(f"Demiryollari: {len(railways_gdf)} geometri, {len(railway_meshes)} mesh")

    if not args.without_water:
        water_polygon_raw = fetch_osm_layer(
            "water_polygons",
            boundary_polygon,
            WATER_POLYGON_TAGS,
            feature_cache_dir,
        )
        water_line_raw = fetch_osm_layer(
            "water_lines",
            boundary_polygon,
            WATER_LINE_TAGS,
            feature_cache_dir,
        )

        water_polygons_gdf, water_polygon_meshes = build_polygon_layer_meshes(
            raw_gdf=water_polygon_raw,
            boundary_polygon=boundary_polygon,
            boundary_projected=boundary_projected,
            terrain_context=terrain_context,
            terrain_embed_m=terrain_embed_m,
            min_area_m2=min_area_m2,
            height_m=water_height_m,
        )
        water_lines_gdf, water_line_meshes = build_buffered_linear_layer_meshes(
            raw_gdf=water_line_raw,
            boundary_polygon=boundary_polygon,
            boundary_projected=boundary_projected,
            terrain_context=terrain_context,
            terrain_embed_m=terrain_embed_m,
            min_area_m2=min_area_m2,
            min_width_m=min_width_m,
            height_m=water_height_m,
            width_resolver=lambda row, fallback: max(
                parse_length_to_meters(row.get("width")) or water_width_from_waterway(row.get("waterway")),
                fallback,
            ),
        )
        layer_meshes.extend(water_polygon_meshes)
        layer_meshes.extend(water_line_meshes)
        layer_counts["water"] = len(water_polygons_gdf) + len(water_lines_gdf)
        print(
            f"Su katmani: {len(water_polygons_gdf) + len(water_lines_gdf)} geometri, "
            f"{len(water_polygon_meshes) + len(water_line_meshes)} mesh"
        )

    if not args.without_parks:
        parks_raw = fetch_osm_layer("parks", boundary_polygon, PARK_TAGS, feature_cache_dir)
        parks_gdf, park_meshes = build_park_layer_meshes(
            raw_gdf=parks_raw,
            boundary_polygon=boundary_polygon,
            boundary_projected=boundary_projected,
            terrain_context=terrain_context,
            print_tuning=print_tuning,
            terrain_embed_m=terrain_embed_m,
            min_area_m2=min_area_m2,
            height_m=park_height_m,
        )
        layer_meshes.extend(park_meshes)
        layer_counts["parks"] = len(parks_gdf)
        print(f"Parklar: {len(parks_gdf)} geometri, {len(park_meshes)} mesh")

    scene = trimesh.util.concatenate(layer_meshes)
    scene.apply_scale(scale_factor)
    scene = finalize_scene_mesh(scene)
    scene.apply_translation(-scene.bounds[0])

    output_path = resolve_output_path(args, label)
    scene.export(output_path)

    bounds = scene.bounds
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"STL hazir: {output_path}")
    print(f"Dosya boyutu: {size_mb:.2f} MB")
    print(
        "Olcek sonrasi boyutlar (mm):",
        {
            "genislik": round(float(bounds[1][0] - bounds[0][0]), 2),
            "derinlik": round(float(bounds[1][1] - bounds[0][1]), 2),
            "yukseklik": round(float(bounds[1][2] - bounds[0][2]), 2),
        },
    )
    print("Katman ozeti:", layer_counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

