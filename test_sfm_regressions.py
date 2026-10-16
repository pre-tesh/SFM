import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np

import config
from geometry.triangulation import triangulate_points, valid_triangulation_mask
from bundle_adjustment import bundle_adjustment, project_point
from ransac import ransac_F, sampson_distance_squared
from geometry.essential import compute_essentialmatrix


class ReconstructionRegressions(unittest.TestCase):
    def setUp(self):
        self.K = config.K
        self.R = np.eye(3)
        self.t = np.array([-1.0, 0.0, 0.0])
        self.points = np.array([[0.1, 0.2, 5.0], [-0.4, 0.3, 7.0], [0.5, -0.1, 6.0]])
        self.P1 = self.K @ np.column_stack((self.R, np.zeros(3)))
        self.P2 = self.K @ np.column_stack((self.R, self.t))
        self.pixels1 = np.array([project_point(p, self.R, np.zeros(3), self.K) for p in self.points])
        self.pixels2 = np.array([project_point(p, self.R, self.t, self.K) for p in self.points])

    def test_signed_homogeneous_coordinates(self):
        homogeneous = np.column_stack((self.points, np.ones(3))).T
        homogeneous[:, 0] *= -1.0
        with patch('geometry.triangulation.cv.triangulatePoints', return_value=homogeneous):
            result = triangulate_points(self.P1, self.P2, self.pixels1, self.pixels2)
        np.testing.assert_allclose(result, self.points)

    def test_points_at_infinity_keep_alignment(self):
        homogeneous = np.column_stack((self.points, np.ones(3))).T
        homogeneous[3, 1] = 0.0
        with patch('geometry.triangulation.cv.triangulatePoints', return_value=homogeneous):
            result = triangulate_points(self.P1, self.P2, self.pixels1, self.pixels2)
        self.assertEqual(result.shape, (3, 3))
        self.assertTrue(np.isnan(result[1]).all())
        np.testing.assert_allclose(result[[0, 2]], self.points[[0, 2]])

    def test_triangulation_reprojects_in_both_cameras(self):
        points = triangulate_points(self.P1, self.P2, self.pixels1, self.pixels2)
        np.testing.assert_allclose(points, self.points, atol=1e-8)
        valid = valid_triangulation_mask(points, self.pixels1, self.pixels2,
                                        self.K, self.R, np.zeros(3), self.R, self.t)
        self.assertTrue(valid.all())
        bad_pixels = self.pixels2.copy()
        bad_pixels[0, 0] += 100
        valid = valid_triangulation_mask(points, self.pixels1, bad_pixels,
                                        self.K, self.R, np.zeros(3), self.R, self.t)
        self.assertFalse(valid[0])

    def test_ransac_rejects_outliers_and_is_scale_invariant(self):
        rng = np.random.default_rng(7)
        points = rng.uniform([-1, -1, 4], [1, 1, 9], (100, 3))
        a = np.array([project_point(p, self.R, np.zeros(3), self.K) for p in points])
        b = np.array([project_point(p, self.R, self.t, self.K) for p in points])
        a += rng.normal(0, 0.2, a.shape)
        b += rng.normal(0, 0.2, b.shape)
        b[:20] = rng.uniform([500, 500], [3000, 4000], (20, 2))
        F, mask = ransac_F(a, b)
        self.assertGreaterEqual(mask[20:].sum(), 75)
        self.assertLessEqual(mask[:20].sum(), 2)
        ah = np.column_stack((a, np.ones(len(a))))
        bh = np.column_stack((b, np.ones(len(b))))
        np.testing.assert_allclose(sampson_distance_squared(F, ah, bh),
                                   sampson_distance_squared(-100 * F, ah, bh), atol=1e-6)
        singular = np.linalg.svd(compute_essentialmatrix(F), compute_uv=False)
        self.assertAlmostEqual(singular[0], singular[1], places=8)
        self.assertLess(singular[2], 1e-10)

    def test_bundle_adjustment_retains_two_view_tracks(self):
        rng = np.random.default_rng(42)
        points = rng.uniform([-0.5, -0.5, 4], [0.5, 0.5, 8], (30, 3))
        poses = {0: (self.R, np.zeros(3)), 1: (self.R, self.t)}
        observations = [(cam, i, project_point(p, R, t, self.K))
                        for cam, (R, t) in poses.items() for i, p in enumerate(points)]
        initial = points + rng.normal(0, 0.002, points.shape)
        with contextlib.redirect_stdout(io.StringIO()) as log:
            optimized_poses, optimized = bundle_adjustment(poses, initial, observations)
        self.assertNotIn('Skipping BA', log.getvalue())
        before = np.array([project_point(initial[i], *poses[cam], self.K) - uv
                           for cam, i, uv in observations])
        after = np.array([project_point(optimized[i], *optimized_poses[cam], self.K) - uv
                          for cam, i, uv in observations])
        self.assertLess(np.linalg.norm(after), np.linalg.norm(before))
        self.assertEqual(optimized.shape, initial.shape)
        self.assertAlmostEqual(np.linalg.norm(optimized_poses[1][1]), 1.0, places=7)


if __name__ == '__main__':
    unittest.main()
