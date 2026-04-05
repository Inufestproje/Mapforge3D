from __future__ import annotations

from main import main as run_main


if __name__ == "__main__":
    raise SystemExit(
        run_main(
            [
                "place-radius",
                "Battalgazi, Malatya, Turkey",
                "500",
                "--without-roads",
                "--without-water",
                "--without-parks",
                "--target-size-mm",
                "140",
            ]
        )
    )
