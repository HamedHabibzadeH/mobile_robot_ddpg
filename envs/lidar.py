# -*- coding: utf-8 -*-
import numpy as np

class Lidar:
    def __init__(self, n_rays, fov_deg, max_range, noise_std=0.0, rng=None):
        self.n = int(n_rays)
        self.fov = np.deg2rad(fov_deg)
        self.max_range = float(max_range)
        self.noise_std = float(noise_std)
        self.rng = np.random.default_rng(rng)

    def scan(self, pose, obstacles, world_inner_aabb):
        """
        pose: (x, y, theta)
        موانع: لیست آبجکت‌هایی با متد ray_intersection_distance
        world_inner_aabb: AABB کاهش‌یافته (مرزهای دیوار از داخل)
        خروجی: بردار فاصله‌ها (نرمال‌شده 0..1)
        """
        x, y, th = pose
        start_angle = th - self.fov*0.5
        # جهت‌ پرتوها
        angles = start_angle + np.linspace(0.0, self.fov, self.n)
        dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)

        dists = np.empty(self.n, dtype=np.float32)
        origin = np.array([x, y], dtype=np.float32)

        for i, d in enumerate(dirs):
            d = d / (np.linalg.norm(d) + 1e-12)
            # فاصله تا دیوارها
            min_dist = world_inner_aabb.ray_intersection_distance(origin, d)
            # فاصله تا هر مانع
            for ob in obstacles:
                di = ob.ray_intersection_distance(origin, d)
                if di < min_dist:
                    min_dist = di
            # برش تا ماکس رنج
            min_dist = np.clip(min_dist, 0.0, self.max_range)
            dists[i] = min_dist / self.max_range

        if self.noise_std > 0.0:
            noise = self.rng.normal(0.0, self.noise_std, size=self.n).astype(np.float32)
            dists = np.clip(dists + noise, 0.0, 1.0)

        return dists
