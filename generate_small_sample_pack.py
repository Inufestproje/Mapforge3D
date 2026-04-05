from __future__ import annotations

import ast
import argparse
import json
import locale
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "3_boyutlu_stl_dosyaları" / "ornek_paket_kucuk_5li"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
PREFERRED_ENCODING = locale.getpreferredencoding(False) or "utf-8"


@dataclass(frozen=True)
class SampleSpec:
    slug: str
    title: str
    concept: str
    lat: float
    lon: float
    radius_m: int
    target_size_mm: float
    terrain_z_scale: float
    note: str

    def output_path(self) -> Path:
        return OUTPUT_DIR / f"{self.slug}.stl"

    def command(self) -> list[str]:
        return [
            sys.executable,
            "main.py",
            "point-radius",
            str(self.lat),
            str(self.lon),
            str(self.radius_m),
            "--print-profile",
            "fdm",
            "--selection",
            "largest",
            "--max-buildings",
            "20",
            "--target-size-mm",
            str(self.target_size_mm),
            "--base-thickness-mm",
            "6.0",
            "--terrain-z-scale",
            str(self.terrain_z_scale),
            "--terrain-zoom",
            "14",
            "--terrain-max-size",
            "220",
            "--terrain-embed-mm",
            "0.45",
            "--output",
            str(self.output_path()),
        ]


SAMPLES: list[SampleSpec] = [
    SampleSpec(
        slug="01_venice_canal_cluster",
        title="Venice Canal Cluster",
        concept="kanal odakli tarihi ada dokusu",
        lat=45.4379,
        lon=12.3358,
        radius_m=170,
        target_size_mm=120.0,
        terrain_z_scale=1.1,
        note="Su kanallari ve sik dokulu tarihi binalar; duz arazide okunakli bir FDM ornegi.",
    ),
    SampleSpec(
        slug="02_santorini_cliff_village",
        title="Santorini Cliff Village",
        concept="yamac uzerine kurulu kiyisal koy",
        lat=36.4619,
        lon=25.3753,
        radius_m=190,
        target_size_mm=118.0,
        terrain_z_scale=2.0,
        note="Dik topografya, teras etkisi ve kompakt bina adalariyla daha organik bir sahne.",
    ),
    SampleSpec(
        slug="03_oxford_college_courtyard",
        title="Oxford College Courtyard",
        concept="kampus ve avlu yerlesimi",
        lat=51.7536,
        lon=-1.2546,
        radius_m=220,
        target_size_mm=122.0,
        terrain_z_scale=1.2,
        note="Buyuk kurumsal kutleler, avlular ve yesil bosluklarla duzenli bir kampus karakteri.",
    ),
    SampleSpec(
        slug="04_barcelona_eixample_blocks",
        title="Barcelona Eixample Blocks",
        concept="duzenli modern blok ve bulvar dokusu",
        lat=41.3927,
        lon=2.1649,
        radius_m=190,
        target_size_mm=118.0,
        terrain_z_scale=1.1,
        note="Pahli koseli bloklar ve genis caddelerle duzenli, net okunan modern bir sahne.",
    ),
    SampleSpec(
        slug="05_kyoto_temple_quarter",
        title="Kyoto Temple Quarter",
        concept="geleneksel dusuk katli mahalle",
        lat=35.0037,
        lon=135.7788,
        radius_m=210,
        target_size_mm=120.0,
        terrain_z_scale=1.5,
        note="Dusuk katli doku, tapinak cevresi ve yumusak egimle sakin bir kent parcasi.",
    ),
]


def parse_line_value(output: str, prefix: str) -> Any | None:
    for line in output.splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    return None


def parse_metrics(output: str) -> dict[str, Any]:
    building_line = parse_line_value(output, "Binalar")
    layer_line = parse_line_value(output, "Katman ozeti")
    size_line = parse_line_value(output, "Olcek sonrasi boyutlar (mm)")
    size_mb_line = parse_line_value(output, "Dosya boyutu")

    building_count = None
    if building_line:
        building_count = int(building_line.split("geometri", 1)[0].strip().split()[-1])

    return {
        "building_count": building_count,
        "layer_counts": ast.literal_eval(layer_line) if layer_line else None,
        "scene_bounds_mm": ast.literal_eval(size_line) if size_line else None,
        "file_size_mb": float(size_mb_line.split()[0]) if size_mb_line else None,
    }


def generate_sample(spec: SampleSpec) -> dict[str, Any]:
    print(f"[olusturuluyor] {spec.slug}")
    result = subprocess.run(
        spec.command(),
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding=PREFERRED_ENCODING,
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{spec.slug} uretilemedi.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    metrics = parse_metrics(result.stdout)
    record = {
        "slug": spec.slug,
        "title": spec.title,
        "concept": spec.concept,
        "note": spec.note,
        "center": {"lat": spec.lat, "lon": spec.lon},
        "radius_m": spec.radius_m,
        "target_size_mm": spec.target_size_mm,
        "terrain_z_scale": spec.terrain_z_scale,
        "print_profile": "fdm",
        "selection": "largest",
        "max_buildings": 20,
        "output_stl": spec.output_path().name,
        **metrics,
    }
    print(
        f"  -> {record['building_count']} bina, "
        f"{record['scene_bounds_mm']}, "
        f"{record['file_size_mb']} MB"
    )
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kucuk ve 3D yazici dostu 5 STL ornek paketi uretir.")
    parser.add_argument("--only", choices=[sample.slug for sample in SAMPLES], default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected_samples = [sample for sample in SAMPLES if args.only in (None, sample.slug)]
    generated_records = {record["slug"]: record for record in (generate_sample(spec) for spec in selected_samples)}

    existing_records: dict[str, dict[str, Any]] = {}
    if MANIFEST_PATH.exists():
        try:
            payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            for record in payload.get("samples", []):
                slug = record.get("slug")
                if isinstance(slug, str):
                    existing_records[slug] = record
        except json.JSONDecodeError:
            existing_records = {}

    merged_records = []
    for sample in SAMPLES:
        if sample.slug in generated_records:
            merged_records.append(generated_records[sample.slug])
        elif sample.slug in existing_records:
            merged_records.append(existing_records[sample.slug])

    manifest = {
        "pack_name": "kucuk_5li_ornek_paket",
        "generated_with": Path(__file__).name,
        "samples": merged_records,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nManifest hazir: {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
