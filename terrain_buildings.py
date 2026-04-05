from __future__ import annotations

from main import main as run_main


if __name__ == "__main__":
    raise SystemExit(
        run_main(
            [
                "point-radius",
                "38.35",
                "38.32",
                "250",
                "--without-roads",
                "--without-water",
                "--without-parks",
            ]
        )
    )
