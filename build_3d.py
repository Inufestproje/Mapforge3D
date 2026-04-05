from __future__ import annotations

from main import main as run_main


if __name__ == "__main__":
    raise SystemExit(
        run_main(
            [
                "bbox",
                "38.357",
                "38.356",
                "38.318",
                "38.317",
                "--without-roads",
                "--without-water",
                "--without-parks",
            ]
        )
    )
