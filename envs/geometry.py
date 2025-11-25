# -*- coding: utf-8 -*-
from dataclasses import dataclass
import numpy as np

@dataclass
class AABB:
    """Axis-Aligned Bounding Box"""
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def contains_point(self, p):
        x, y = p
        return (self.xmin <= x <= self.xmax) and (self.ymin <= y <= self.ymax)

    def distance_to_point(self, p):
        x, y = p
        dx = max(self.xmin - x, 0, x - self.xmax)
        dy = max(self.ymin - y, 0, y - self.ymax)
        return np.hypot(dx, dy)

    def ray_intersection_distance(self, origin, direction, inside_ok=True):
        """
        محاسبه‌ی فاصله‌ی برخورد پرتو با AABB
        اگر مبدا داخل AABB باشد، اولین خروجی (tmax مثبت) برگردانده می‌شود.
        """
        ox, oy = origin
        dx, dy = direction
        inv_dx = 1.0 / dx if abs(dx) > 1e-12 else np.inf
        inv_dy = 1.0 / dy if abs(dy) > 1e-12 else np.inf

        t1 = (self.xmin - ox) * inv_dx
        t2 = (self.xmax - ox) * inv_dx
        t3 = (self.ymin - oy) * inv_dy
        t4 = (self.ymax - oy) * inv_dy

        tmin = max(min(t1, t2), min(t3, t4))
        tmax = min(max(t1, t2), max(t3, t4))

        if tmax < 0:       # کل AABB پشت سر پرتو
            return np.inf
        if tmin > tmax:    # بدون برخورد
            return np.inf

        # اگر داخل هستیم، اولین خروجی
        if self.contains_point(origin) and inside_ok:
            return tmax if tmax >= 0 else np.inf
        # در غیر این صورت اولین ورود
        return tmin if tmin >= 0 else (tmax if tmax >= 0 else np.inf)


@dataclass
class CircleObstacle:
    cx: float
    cy: float
    r: float   # شعاع «بادکرده» (شامل شعاع ربات)

    def collides(self, p, r_robot=0.0):
        return ( (p[0]-self.cx)**2 + (p[1]-self.cy)**2 ) <= (self.r)**2

    def ray_intersection_distance(self, origin, direction):
        # حل تحلیلی پرتو-دایره
        ox, oy = origin
        dx, dy = direction  # فرض: نرمالیزه
        ocx, ocy = ox - self.cx, oy - self.cy
        b = 2.0 * (dx*ocx + dy*ocy)
        c = ocx*ocx + ocy*ocy - self.r*self.r
        disc = b*b - 4.0*c
        if disc < 0:
            return np.inf
        sqrt_disc = np.sqrt(disc)
        t1 = (-b - sqrt_disc) / 2.0
        t2 = (-b + sqrt_disc) / 2.0
        # اولین t مثبت
        ts = [t for t in (t1, t2) if t >= 0.0]
        return min(ts) if ts else np.inf


@dataclass
class BoxObstacle:
    cx: float
    cy: float
    w: float
    h: float
    inflate: float   # بادکردگی (برای درنظر گرفتن شعاع ربات)

    def aabb(self) -> AABB:
        half_w = self.w*0.5 + self.inflate
        half_h = self.h*0.5 + self.inflate
        return AABB(self.cx-half_w, self.cy-half_h, self.cx+half_w, self.cy+half_h)

    def collides(self, p, r_robot=0.0):
        aabb = self.aabb()
        return aabb.contains_point(p)

    def ray_intersection_distance(self, origin, direction):
        return self.aabb().ray_intersection_distance(origin, direction, inside_ok=False)
