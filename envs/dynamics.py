# -*- coding: utf-8 -*-
from dataclasses import dataclass
import numpy as np

@dataclass
class RobotParams:
    dt: float
    v_min: float
    v_max: float
    w_min: float
    w_max: float
    a_v_max: float
    a_w_max: float
    radius: float

class UnicycleDynamics:
    """
    اکشن = [a_v, a_w]
    حالت ربات: [x, y, theta, v, w]
    """
    def __init__(self, params: RobotParams):
        self.p = params
        self.state = None

    def reset(self, x, y, theta, v=0.0, w=0.0):
        self.state = np.array([x, y, theta, v, w], dtype=np.float32)
        return self.state.copy()

    def step(self, action):
        assert self.state is not None
        dt = self.p.dt
        a_v = float(np.clip(action[0], -1.0, 1.0)) * self.p.a_v_max
        a_w = float(np.clip(action[1], -1.0, 1.0)) * self.p.a_w_max

        x, y, th, v, w = self.state

        # آپدیت سرعت‌ها
        v = np.clip(v + a_v*dt, self.p.v_min, self.p.v_max)
        w = np.clip(w + a_w*dt, self.p.w_min, self.p.w_max)

        # آپدیت وضعیت
        x = x + v*np.cos(th)*dt
        y = y + v*np.sin(th)*dt
        th = (th + w*dt + np.pi) % (2*np.pi) - np.pi  # نرمال‌سازی زاویه

        self.state[:] = [x, y, th, v, w]
        return self.state.copy(), np.array([a_v, a_w], dtype=np.float32)
