from __future__ import annotations

import unittest

from utils import (
    estimate_railway_surface_width,
    estimate_visible_transport_layer,
    parse_railway_gauge_m,
)


class RailwayUtilsTests(unittest.TestCase):
    def test_parse_railway_gauge_defaults_to_millimeters(self) -> None:
        self.assertAlmostEqual(parse_railway_gauge_m("1435"), 1.435, places=3)
        self.assertAlmostEqual(parse_railway_gauge_m("1000 mm"), 1.0, places=3)

    def test_estimate_railway_surface_width_uses_explicit_width(self) -> None:
        row = {"railway": "rail", "gauge": "1435", "tracks": "2", "width": "9 m"}
        width = estimate_railway_surface_width(row, fallback_width_m=1.0)
        self.assertGreaterEqual(width, 9.0)

    def test_estimate_railway_surface_width_expands_for_multiple_tracks(self) -> None:
        row = {"railway": "rail", "gauge": "1435", "tracks": "2"}
        width = estimate_railway_surface_width(row, fallback_width_m=1.0)
        self.assertGreater(width, 7.5)

    def test_estimate_visible_transport_layer_hides_tunnels(self) -> None:
        self.assertIsNone(estimate_visible_transport_layer({"tunnel": "yes"}))
        self.assertEqual(estimate_visible_transport_layer({"bridge": "yes"}), 1.0)
        self.assertIsNone(estimate_visible_transport_layer({"layer": "-1"}))


if __name__ == "__main__":
    unittest.main()
