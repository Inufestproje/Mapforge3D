from __future__ import annotations

import unittest

from shapely.geometry import Polygon

from utils import create_surface_following_prism_meshes


class SurfaceFollowingPrismTests(unittest.TestCase):
    def test_prism_follows_surface_at_each_vertex(self) -> None:
        polygon = Polygon([(0.0, 0.0), (8.0, 0.0), (8.0, 3.0), (0.0, 3.0)])
        bottom_offset = -0.4
        top_offset = 1.6

        def surface_z(x: float, y: float) -> float:
            return x * 0.35 + y * 0.15

        meshes = create_surface_following_prism_meshes(
            polygon,
            bottom_surface_z_resolver=lambda x, y: surface_z(x, y) + bottom_offset,
            top_surface_z_resolver=lambda x, y: surface_z(x, y) + top_offset,
        )

        self.assertEqual(len(meshes), 1)
        mesh = meshes[0]
        self.assertTrue(mesh.is_watertight)

        z_values_by_xy: dict[tuple[float, float], set[float]] = {}
        for x, y, z in mesh.vertices:
            key = (round(float(x), 6), round(float(y), 6))
            z_values_by_xy.setdefault(key, set()).add(round(float(z), 6))

        self.assertGreaterEqual(len(z_values_by_xy), 4)
        for (x, y), z_values in z_values_by_xy.items():
            self.assertEqual(len(z_values), 2)
            expected = {
                round(surface_z(x, y) + bottom_offset, 6),
                round(surface_z(x, y) + top_offset, 6),
            }
            self.assertEqual(z_values, expected)


if __name__ == "__main__":
    unittest.main()
