# # -*- coding: utf-8 -*-
# import math
# import time
# from dataclasses import dataclass
# from typing import List, Tuple

# import numpy as np
# import gymnasium as gym
# from gymnasium import spaces
# import matplotlib.pyplot as plt

# from .geometry import AABB, CircleObstacle, BoxObstacle
# from .lidar import Lidar
# from .dynamics import RobotParams, UnicycleDynamics


# # ===================== پیکربندی =====================
# @dataclass
# class WorldConfig:
#     width: float
#     height: float
#     robot_radius: float
#     lidar_n: int
#     lidar_fov_deg: float
#     lidar_max_range: float
#     lidar_noise_std: float
#     dt: float
#     v_limits: Tuple[float, float]
#     w_limits: Tuple[float, float]
#     a_limits: Tuple[float, float]
#     goal_radius: float
#     safe_distance: float
#     max_steps: int
#     n_circles: int
#     n_boxes: int
#     min_gap: float
#     obstacle_tries: int
#     seed: int


# # ===================== مانع متحرک =====================
# from .geometry import CircleObstacle, AABB

# class MovingCircleObstacle(CircleObstacle):
#     """
#     مانع دایره‌ای متحرک بین دو نقطه (x_min, x_max) روی محور x.
#     از CircleObstacle ارث‌بری می‌کند تا با LiDAR سازگار باشد.
#     """
#     def __init__(self, cx, cy, r, x_min, x_max, speed):
#         super().__init__(cx, cy, r)  # ← حالا همه متدهای هندسی از جمله ray_intersection_distance را دارد
#         self.x_min = x_min
#         self.x_max = x_max
#         self.speed = speed
#         self.direction = 1  # +1 راست، -1 چپ

#     def update(self, dt):
#         self.cx += self.direction * self.speed * dt
#         if self.cx >= self.x_max:
#             self.cx = self.x_max
#             self.direction = -1
#         elif self.cx <= self.x_min:
#             self.cx = self.x_min
#             self.direction = 1


# # ===================== محیط =====================
# class MobileRobotNavEnv(gym.Env):
#     metadata = {
#         "render_modes": ["human", "none"],
#         "render_fps": 1.0/0.08
#     }

#     def __init__(self, cfg: WorldConfig, reward_cfg: dict):
#         super().__init__()
#         self.cfg = cfg
#         self.rw = reward_cfg
#         self.rng = np.random.default_rng(cfg.seed)

#         # دیواره‌ها
#         self.world_aabb = AABB(
#             cfg.robot_radius,
#             cfg.robot_radius,
#             cfg.width - cfg.robot_radius,
#             cfg.height - cfg.robot_radius,
#         )

#         # دینامیک
#         self.params = RobotParams(
#             dt=cfg.dt,
#             v_min=cfg.v_limits[0], v_max=cfg.v_limits[1],
#             w_min=cfg.w_limits[0], w_max=cfg.w_limits[1],
#             a_v_max=cfg.a_limits[0], a_w_max=cfg.a_limits[1],
#             radius=cfg.robot_radius
#         )
#         self.model = UnicycleDynamics(self.params)

#         # LiDAR
#         self.lidar = Lidar(cfg.lidar_n, cfg.lidar_fov_deg, cfg.lidar_max_range,
#                            noise_std=cfg.lidar_noise_std, rng=self.rng)

#         # فضاهای اکشن و مشاهده
#         obs_dim = cfg.lidar_n + 1 + 2 + 2 + 1
#         self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
#         self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

#         # متغیرهای داخلی
#         self.obstacles: List = []
#         self.moving_obstacle = None
#         self.goal_xy = np.zeros(2, dtype=np.float32)
#         self.prev_goal_dist = None
#         self.prev_action = np.zeros(2, dtype=np.float32)
#         self.steps = 0
#         self.render_fig = None
#         self.render_ax = None
#         self.trajectory = []

#     # ---------- تولید موانع ----------
#     def _generate_obstacles(self):
#         self.obstacles.clear()

#         # --- مانع متحرک (دایره‌ای رفت و برگشتی) ---
#         mid_y = self.cfg.height * 0.5
#         moving_ob = MovingCircleObstacle(
#             cx=self.cfg.width * 0.3,
#             cy=mid_y,
#             r=0.4 + self.cfg.robot_radius,
#             x_min=self.cfg.width * 0.25,
#             x_max=self.cfg.width * 0.75,
#             speed=2.0  # واحد بر ثانیه
#         )
#         self.moving_obstacle = moving_ob
#         self.obstacles.append(moving_ob)

#         # --- مانع ثابت ---
#         cx = self.cfg.width * 0.8
#         cy = self.cfg.height * 0.25
#         ob = CircleObstacle(cx, cy, 0.4 + self.cfg.robot_radius)
#         self.obstacles.append(ob)

#     # ---------- Gym API ----------
#     def reset(self, *, seed=None, options=None):
#         if seed is not None:
#             self.rng = np.random.default_rng(seed)

#         self._generate_obstacles()

#         # موقعیت شروع و هدف طوری که ربات مجبور به عبور از مانع متحرک باشد
#         start = np.array([self.cfg.width * 0.15, self.cfg.height * 0.5])
#         goal = np.array([self.cfg.width * 0.9, self.cfg.height * 0.5])

#         th0 = 0.0  # رو به راست
#         self.model.reset(start[0], start[1], th0, v=0.0, w=0.0)
#         self.goal_xy[:] = goal
#         self.prev_goal_dist = np.linalg.norm(self.goal_xy - start)
#         self.prev_action[:] = 0.0
#         self.steps = 0
#         self.trajectory = [start.copy()]

#         obs = self._get_obs()
#         info = {"is_success": False}
#         return obs, info

#     def step(self, action):
#         self.steps += 1

#         # --- به‌روزرسانی مانع متحرک ---
#         if self.moving_obstacle is not None:
#             self.moving_obstacle.update(self.cfg.dt)

#         # دینامیک ربات
#         state_before = self.model.state.copy()
#         s, a_real = self.model.step(action)
#         x, y, th, v, w = s
#         self.trajectory.append(np.array([x, y], dtype=np.float32))

#         # چک برخورد و مرز
#         out = not self.world_aabb.contains_point((x, y))
#         collision = self._check_collision((x, y))

#         # پاداش
#         reward, done_success, done_collision, done_timeout = self._compute_reward(
#             state_before, s, a_real, collision, out
#         )

#         terminated = collision or done_success or out
#         truncated = (self.steps >= self.cfg.max_steps) or done_timeout

#         info = {
#             "is_success": bool(done_success),
#             "collision": bool(collision),
#             "out_of_bounds": bool(out)
#         }
#         obs = self._get_obs()
#         return obs, reward, terminated, truncated, info

#     # ---------- توابع کمکی ----------
#     def _check_collision(self, p):
#         for ob in self.obstacles:
#             if ob.collides(p):
#                 return True
#         return False

#     def _goal_distance(self, p):
#         return np.linalg.norm(self.goal_xy - np.array(p[:2]))

#     def _get_obs(self):
#         x, y, th, v, w = self.model.state
#         lidar = self.lidar.scan((x, y, th), self.obstacles, self.world_aabb)
#         lidar_min = float(np.min(lidar))

#         dx, dy = self.goal_xy[0]-x, self.goal_xy[1]-y
#         ang_to_goal = math.atan2(dy, dx) - th
#         ang_to_goal = (ang_to_goal + np.pi) % (2*np.pi) - np.pi

#         world_diag = math.hypot(self.cfg.width, self.cfg.height)
#         d_goal_norm = np.clip(np.hypot(dx, dy) / world_diag, 0.0, 1.0)
#         v_norm = np.clip((v - self.cfg.v_limits[0]) / max(1e-6, (self.cfg.v_limits[1]-self.cfg.v_limits[0])) * 2 - 1, -1, 1)
#         w_norm = np.clip((w - self.cfg.w_limits[0]) / max(1e-6, (self.cfg.w_limits[1]-self.cfg.w_limits[0])) * 2 - 1, -1, 1)

#         obs = np.concatenate([
#             lidar.astype(np.float32),
#             np.array([d_goal_norm], dtype=np.float32),
#             np.array([np.sin(ang_to_goal), np.cos(ang_to_goal)], dtype=np.float32),
#             np.array([v_norm, w_norm], dtype=np.float32),
#             np.array([lidar_min], dtype=np.float32)
#         ], axis=0)

#         return obs

#     def _compute_reward(self, s_prev, s_curr, a_real, collision, out_of_bounds):
#         x, y, th, v, w = s_curr
#         d_now = self._goal_distance((x, y))
#         progress = (self.prev_goal_dist - d_now)
#         self.prev_goal_dist = d_now

#         r = 0.0
#         r += self.rw["K_PROGRESS"] * progress

#         dx, dy = self.goal_xy[0]-x, self.goal_xy[1]-y
#         ang_to_goal = math.atan2(dy, dx) - th
#         ang_to_goal = (ang_to_goal + np.pi) % (2*np.pi) - np.pi
#         r += self.rw["K_HEADING"] * math.cos(ang_to_goal)

#         r += self.rw["K_TIME"]

#         av, aw = a_real
#         r += self.rw["K_SMOOTH_A"] * (abs(av)/self.cfg.a_limits[0] + abs(aw)/self.cfg.a_limits[1])
#         r += self.rw["K_TURNING"] * (abs(w)/self.cfg.w_limits[1])

#         delta_u = np.abs(np.clip(self.prev_action, -1, 1) -
#                          np.clip(np.array([av/self.cfg.a_limits[0], aw/self.cfg.a_limits[1]]), -1, 1))
#         r += self.rw["K_JERK"] * float(delta_u.mean())
#         self.prev_action = np.array([av/self.cfg.a_limits[0], aw/self.cfg.a_limits[1]], dtype=np.float32)

#         lidar = self.lidar.scan((x, y, th), self.obstacles, self.world_aabb)
#         d_min = float(np.min(lidar)) * self.cfg.lidar_max_range
#         if d_min < self.cfg.safe_distance:
#             r += self.rw["K_CLEARANCE"] * ((self.cfg.safe_distance - d_min) / self.cfg.safe_distance)

#         dist_to_wall = self.world_aabb.distance_to_point((x, y))
#         if dist_to_wall < (self.cfg.robot_radius + 0.1):
#             r += self.rw["K_BOUNDARY"] * ((self.cfg.robot_radius + 0.1 - dist_to_wall) / (self.cfg.robot_radius + 0.1))

#         done_collision = False
#         if collision:
#             r += self.rw["R_COLLISION"]
#             done_collision = True

#         if out_of_bounds:
#             r += self.rw["R_COLLISION"] * 0.5
#             done_collision = True

#         done_success = False
#         if d_now <= self.cfg.goal_radius:
#             r += self.rw["R_GOAL"]
#             done_success = True

#         done_timeout = False

#         return float(r), done_success, done_collision, done_timeout

#     # ---------- رندر ----------
#     def render(self):
#         if self.render_fig is None:
#             self.render_fig, self.render_ax = plt.subplots(figsize=(8, 5))
#             plt.ion(); plt.show(block=False)

#         ax = self.render_ax
#         ax.clear()
#         ax.set_aspect('equal')
#         ax.set_xlim(0, self.cfg.width)
#         ax.set_ylim(0, self.cfg.height)
#         ax.set_title(f"Step {self.steps}")

#         # دیواره‌ها
#         ax.plot([0, self.cfg.width, self.cfg.width, 0, 0],
#                 [0, 0, self.cfg.height, self.cfg.height, 0], lw=2)

#         # موانع
#         for ob in self.obstacles:
#             color = 'red' if ob is self.moving_obstacle else 'gray'
#             c = plt.Circle((ob.cx, ob.cy), ob.r, color=color, fill=True, alpha=0.3)
#             ax.add_patch(c)

#         # هدف و ربات
#         ax.add_patch(plt.Circle((self.goal_xy[0], self.goal_xy[1]), self.cfg.goal_radius, color='green', alpha=0.5))
#         x, y, th, v, w = self.model.state
#         ax.add_patch(plt.Circle((x, y), self.cfg.robot_radius, color='blue', alpha=0.9))
#         ax.arrow(x, y, 0.5*np.cos(th), 0.5*np.sin(th), head_width=0.15, length_includes_head=True)

#         # مسیر
#         if len(self.trajectory) >= 2:
#             traj = np.vstack(self.trajectory)
#             ax.plot(traj[:, 0], traj[:, 1], lw=2)

#         self.render_fig.canvas.draw()
#         self.render_fig.canvas.flush_events()
#         time.sleep(self.cfg.dt)

#     def close(self):
#         if self.render_fig is not None:
#             plt.close(self.render_fig)
#             self.render_fig = None
#             self.render_ax = None


# # ---------- سازنده میان‌بر ----------
# def make_env_from_config(cfg_module):
#     cfg = WorldConfig(
#         width=cfg_module.WORLD_WIDTH,
#         height=cfg_module.WORLD_HEIGHT,
#         robot_radius=cfg_module.ROBOT_RADIUS,
#         lidar_n=cfg_module.LIDAR_N_RAYS,
#         lidar_fov_deg=cfg_module.LIDAR_FOV_DEG,
#         lidar_max_range=cfg_module.LIDAR_MAX_RANGE,
#         lidar_noise_std=cfg_module.LIDAR_NOISE_STD,
#         dt=cfg_module.DT,
#         v_limits=(cfg_module.V_MIN, cfg_module.V_MAX),
#         w_limits=(cfg_module.W_MIN, cfg_module.W_MAX),
#         a_limits=(cfg_module.A_V_MAX, cfg_module.A_W_MAX),
#         goal_radius=cfg_module.GOAL_RADIUS,
#         safe_distance=cfg_module.SAFE_DISTANCE,
#         max_steps=cfg_module.MAX_STEPS,
#         n_circles=cfg_module.N_CIRCLES,
#         n_boxes=cfg_module.N_BOXES,
#         min_gap=cfg_module.MIN_GAP,
#         obstacle_tries=cfg_module.OBSTACLE_TRIES,
#         seed=cfg_module.N_SEED
#     )

#     reward_cfg = dict(
#         R_COLLISION=cfg_module.R_COLLISION,
#         R_GOAL=cfg_module.R_GOAL,
#         K_PROGRESS=cfg_module.K_PROGRESS,
#         K_HEADING=cfg_module.K_HEADING,
#         K_TIME=cfg_module.K_TIME,
#         K_TURNING=cfg_module.K_TURNING,
#         K_SMOOTH_A=cfg_module.K_SMOOTH_A,
#         K_JERK=cfg_module.K_JERK,
#         K_CLEARANCE=cfg_module.K_CLEARANCE,
#         K_BOUNDARY=cfg_module.K_BOUNDARY,
#     )
#     return MobileRobotNavEnv(cfg, reward_cfg)

# -*- coding: utf-8 -*-
import math
import time
from dataclasses import dataclass
from typing import List, Tuple


import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt


from .geometry import AABB, CircleObstacle, BoxObstacle
from .lidar import Lidar
from .dynamics import RobotParams, UnicycleDynamics


@dataclass
class WorldConfig:
    width: float
    height: float
    robot_radius: float
    lidar_n: int
    lidar_fov_deg: float
    lidar_max_range: float
    lidar_noise_std: float
    dt: float
    v_limits: Tuple[float, float]
    w_limits: Tuple[float, float]
    a_limits: Tuple[float, float]
    goal_radius: float
    safe_distance: float
    max_steps: int
    n_circles: int
    n_boxes: int
    min_gap: float
    obstacle_tries: int
    seed: int




class MobileRobotNavEnv(gym.Env):
    """
    مشاهده‌ها:
      - بردار LiDAR (N مقدار در 0..1)
      - d_goal_norm (0..1), sin(theta_to_goal), cos(theta_to_goal)
      - v_norm (-1..1), w_norm (-1..1)
      - min_lidar (0..1)
    اکشن:
      - [a_v_norm, a_w_norm] در بازه [-1, 1]
    """
    metadata = {
        "render_modes": ["human", "none"],
        "render_fps": 1.0/0.08
    }


    def __init__(self, cfg: WorldConfig, reward_cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.rw = reward_cfg
        self.rng = np.random.default_rng(cfg.seed)


        # AABB دیواره‌های داخلی (با کم کردن شعاع ربات)
        self.world_aabb = AABB(
            cfg.robot_radius,
            cfg.robot_radius,
            cfg.width - cfg.robot_radius,
            cfg.height - cfg.robot_radius,
        )


        # دینامیک
        self.params = RobotParams(
            dt=cfg.dt,
            v_min=cfg.v_limits[0], v_max=cfg.v_limits[1],
            w_min=cfg.w_limits[0], w_max=cfg.w_limits[1],
            a_v_max=cfg.a_limits[0], a_w_max=cfg.a_limits[1],
            radius=cfg.robot_radius
        )
        self.model = UnicycleDynamics(self.params)


        # LiDAR
        self.lidar = Lidar(cfg.lidar_n, cfg.lidar_fov_deg, cfg.lidar_max_range,
                           noise_std=cfg.lidar_noise_std, rng=self.rng)


        # فضاهای اکشن و مشاهده
        obs_dim = cfg.lidar_n + 1 + 2 + 2 + 1  # lidar + d + sin/cos + v/w + min_lidar
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)


        # متغیرهای داخلی
        self.obstacles: List = []
        self.goal_xy = np.zeros(2, dtype=np.float32)
        self.prev_goal_dist = None
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.steps = 0
        self.render_fig = None
        self.render_ax = None
        self.trajectory = []


    # ---------- تولید موانع و نمونه اولیه ----------
    def _sample_noncolliding_point(self, margin=0.5):
        for _ in range(128):
            x = self.rng.uniform(self.world_aabb.xmin+margin, self.world_aabb.xmax-margin)
            y = self.rng.uniform(self.world_aabb.ymin+margin, self.world_aabb.ymax-margin)
            p = np.array([x, y], dtype=np.float32)
            if not self._point_in_obstacles(p, self.cfg.robot_radius + self.cfg.min_gap*0.5):
                return p
        return p  # در بدترین حالت


    def _point_in_obstacles(self, p, inflate=0.0):
        # بررسی برخورد نقطه (دایره با شعاع inflate) با موانع
        for ob in self.obstacles:
            if isinstance(ob, CircleObstacle):
                if ((p[0]-ob.cx)**2 + (p[1]-ob.cy)**2) <= (ob.r + inflate)**2:
                    return True
            else:
                aabb = ob.aabb()
                aabb_inf = AABB(aabb.xmin-inflate, aabb.ymin-inflate,
                                aabb.xmax+inflate, aabb.ymax+inflate)
                if aabb_inf.contains_point(p):
                    return True
        return False


    def _generate_obstacles(self):
        self.obstacles.clear()
        tries = self.cfg.obstacle_tries
        # دایره‌ها
        for _ in range(self.cfg.n_circles):
            for _ in range(tries):
                r = self.rng.uniform(0.20, 0.50)
                cx = self.rng.uniform(self.world_aabb.xmin+0.5, self.world_aabb.xmax-0.5)
                cy = self.rng.uniform(self.world_aabb.ymin+0.5, self.world_aabb.ymax-0.5)
                ob = CircleObstacle(cx, cy, r + self.cfg.robot_radius)  # باد شده
                # فاصله با سایر موانع
                ok = True
                for o2 in self.obstacles:
                    if isinstance(o2, CircleObstacle):
                        d = math.hypot(cx-o2.cx, cy-o2.cy)
                        if d < (ob.r + o2.r + self.cfg.min_gap):
                            ok = False; break
                    else:
                        if o2.aabb().contains_point((cx, cy)):
                            ok = False; break
                if ok:
                    self.obstacles.append(ob)
                    break


        # باکس‌ها
        for _ in range(self.cfg.n_boxes):
            for _ in range(tries):
                w = self.rng.uniform(0.6, 1.5)
                h = self.rng.uniform(0.6, 1.5)
                cx = self.rng.uniform(self.world_aabb.xmin+0.5, self.world_aabb.xmax-0.5)
                cy = self.rng.uniform(self.world_aabb.ymin+0.5, self.world_aabb.ymax-0.5)
                ob = BoxObstacle(cx, cy, w, h, inflate=self.cfg.robot_radius)
                aabb = ob.aabb()
                # اگر بیرون مرز نباشد و روی موانع قبلی نیفتد
                if (aabb.xmin > self.world_aabb.xmin and aabb.xmax < self.world_aabb.xmax
                    and aabb.ymin > self.world_aabb.ymin and aabb.ymax < self.world_aabb.ymax):
                    overlap = False
                    for o2 in self.obstacles:
                        if isinstance(o2, CircleObstacle):
                            # فاصله مرکز تا AABB
                            dx = max(o2.cx - aabb.xmax, aabb.xmin - o2.cx, 0)
                            dy = max(o2.cy - aabb.ymax, aabb.ymin - o2.cy, 0)
                            if math.hypot(dx, dy) < (o2.r + self.cfg.min_gap):
                                overlap = True; break
                        else:
                            a2 = o2.aabb()
                            if not (aabb.xmax + self.cfg.min_gap < a2.xmin or
                                    aabb.xmin - self.cfg.min_gap > a2.xmax or
                                    aabb.ymax + self.cfg.min_gap < a2.ymin or
                                    aabb.ymin - self.cfg.min_gap > a2.ymax):
                                overlap = True; break
                    if not overlap:
                        self.obstacles.append(ob)
                        break


    # ---------- Gym API ----------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)


        self._generate_obstacles()


        # نمونه‌گیری نقطه شروع و هدف با فاصله‌ی زیاد
        for _ in range(128):
            start = self._sample_noncolliding_point(margin=0.6)
            goal  = self._sample_noncolliding_point(margin=0.6)
            if np.linalg.norm(goal - start) > 0.5*math.hypot(self.cfg.width, self.cfg.height)*0.4:
                break


        th0 = self.rng.uniform(-np.pi, np.pi)
        self.model.reset(start[0], start[1], th0, v=0.0, w=0.0)
        self.goal_xy[:] = goal
        self.prev_goal_dist = np.linalg.norm(self.goal_xy - start)
        self.prev_action[:] = 0.0
        self.steps = 0
        self.trajectory = [start.copy()]


        obs = self._get_obs()
        info = {"is_success": False}
        return obs, info


    def step(self, action):
        self.steps += 1
        # دینامیک
        state_before = self.model.state.copy()
        s, a_real = self.model.step(action)
        x, y, th, v, w = s
        self.trajectory.append(np.array([x, y], dtype=np.float32))


        # چک مرز
        out = not self.world_aabb.contains_point((x, y))
        # برخورد با موانع
        collision = self._check_collision((x, y))


        # ریوارد
        reward, done_success, done_collision, done_timeout = self._compute_reward(state_before, s, a_real, collision, out)


        # خاتمه
        terminated = collision or done_success or out
        truncated  = (self.steps >= self.cfg.max_steps) or done_timeout


        info = {
            "is_success": bool(done_success),
            "collision": bool(collision),
            "out_of_bounds": bool(out)
        }
        obs = self._get_obs()
        return obs, reward, terminated, truncated, info


    # ---------- اجزای کمکی ----------
    def _check_collision(self, p):
        for ob in self.obstacles:
            if isinstance(ob, CircleObstacle):
                if ob.collides(p):
                    return True
            else:
                if ob.collides(p):
                    return True
        return False


    def _goal_distance(self, p):
        return np.linalg.norm(self.goal_xy - np.array(p[:2]))


    def _get_obs(self):
        x, y, th, v, w = self.model.state
        # LiDAR
        lidar = self.lidar.scan((x, y, th), self.obstacles, self.world_aabb)
        lidar_min = float(np.min(lidar))


        # هدف در چارچوب بدنه
        dx, dy = self.goal_xy[0]-x, self.goal_xy[1]-y
        ang_to_goal = math.atan2(dy, dx) - th
        ang_to_goal = (ang_to_goal + np.pi) % (2*np.pi) - np.pi


        # نرمال‌سازی ویژگی‌ها
        world_diag = math.hypot(self.cfg.width, self.cfg.height)
        d_goal_norm = np.clip(np.hypot(dx, dy) / world_diag, 0.0, 1.0)
        v_norm = np.clip((v - self.cfg.v_limits[0]) / max(1e-6, (self.cfg.v_limits[1]-self.cfg.v_limits[0])) * 2 - 1, -1, 1)
        w_norm = np.clip((w - self.cfg.w_limits[0]) / max(1e-6, (self.cfg.w_limits[1]-self.cfg.w_limits[0])) * 2 - 1, -1, 1)


        obs = np.concatenate([
            lidar.astype(np.float32),                  # 0..1
            np.array([d_goal_norm], dtype=np.float32),# 0..1
            np.array([np.sin(ang_to_goal), np.cos(ang_to_goal)], dtype=np.float32), # -1..1
            np.array([v_norm, w_norm], dtype=np.float32),                           # -1..1
            np.array([lidar_min], dtype=np.float32)    # 0..1
        ], axis=0)


        return obs


    def _compute_reward(self, s_prev, s_curr, a_real, collision, out_of_bounds):
        x, y, th, v, w = s_curr
        d_now = self._goal_distance((x, y))
        progress = (self.prev_goal_dist - d_now)
        self.prev_goal_dist = d_now


        # مولفه‌های ریوارد
        r = 0.0
        # 1) پیشرفت به سمت هدف (potential-based shaping)
        r += self.rw["K_PROGRESS"] * progress


        # 2) تراز شدن با هدف
        dx, dy = self.goal_xy[0]-x, self.goal_xy[1]-y
        ang_to_goal = math.atan2(dy, dx) - th
        ang_to_goal = (ang_to_goal + np.pi) % (2*np.pi) - np.pi
        r += self.rw["K_HEADING"] * math.cos(ang_to_goal)  # [-1,1]


        # 3) جریمه زمان
        r += self.rw["K_TIME"]


        # 4) هموار بودن (جریمه شتاب بزرگ)
        av, aw = a_real
        r += self.rw["K_SMOOTH_A"] * (abs(av)/self.cfg.a_limits[0] + abs(aw)/self.cfg.a_limits[1])


        # 5) جریمه پیچیدن زیاد (کم‌پیچ‌وخم)
        r += self.rw["K_TURNING"] * (abs(w)/self.cfg.w_limits[1])


        # 6) جِرک (تغییر ناگهانی فرمان)
        # تغییر اکشن نرمال‌شده
        delta_u = np.abs(np.clip(self.prev_action, -1, 1) - np.clip(np.array([av/self.cfg.a_limits[0], aw/self.cfg.a_limits[1]]), -1, 1))
        r += self.rw["K_JERK"] * float(delta_u.mean())
        self.prev_action = np.array([av/self.cfg.a_limits[0], aw/self.cfg.a_limits[1]], dtype=np.float32)


        # 7) فاصله ایمن از موانع
        lidar = self.lidar.scan((x, y, th), self.obstacles, self.world_aabb)
        d_min = float(np.min(lidar)) * self.cfg.lidar_max_range
        if d_min < self.cfg.safe_distance:
            r += self.rw["K_CLEARANCE"] * ( (self.cfg.safe_distance - d_min) / self.cfg.safe_distance )


        # 8) نزدیک شدن به دیواره‌ها
        dist_to_wall = self.world_aabb.distance_to_point((x, y))
        if dist_to_wall < (self.cfg.robot_radius + 0.1):
            r += self.rw["K_BOUNDARY"] * ( (self.cfg.robot_radius + 0.1 - dist_to_wall) / (self.cfg.robot_radius + 0.1) )


        # 9) برخورد/خروج
        done_collision = False
        if collision:
            r += self.rw["R_COLLISION"]
            done_collision = True


        if out_of_bounds:
            r += self.rw["R_COLLISION"] * 0.5
            done_collision = True


        # 10) رسیدن به هدف
        done_success = False
        if d_now <= self.cfg.goal_radius:
            r += self.rw["R_GOAL"]
            done_success = True


        # 11) گیر کردن/کُندی زیاد (اختیاری: اگر سرعت خیلی کم شد چند گام پشت‌سرهم)
        done_timeout = False  # قابل توسعه


        return float(r), done_success, done_collision, done_timeout


    # ---------- رندر ----------
    def render(self):
        if self.render_fig is None:
            self.render_fig, self.render_ax = plt.subplots(figsize=(8, 5))
            plt.ion(); plt.show(block=False)


        ax = self.render_ax
        ax.clear()
        ax.set_aspect('equal')
        ax.set_xlim(0, self.cfg.width)
        ax.set_ylim(0, self.cfg.height)
        ax.set_title(f"Step {self.steps}")


        # دیواره‌ها
        ax.plot([0, self.cfg.width, self.cfg.width, 0, 0],
                [0, 0, self.cfg.height, self.cfg.height, 0], lw=2)


        # موانع
        for ob in self.obstacles:
            if isinstance(ob, CircleObstacle):
                c = plt.Circle((ob.cx, ob.cy), ob.r, fill=True, alpha=0.25)
                ax.add_patch(c)
            else:
                a = ob.aabb()
                rect = plt.Rectangle((a.xmin, a.ymin), a.xmax-a.xmin, a.ymax-a.ymin, fill=True, alpha=0.25)
                ax.add_patch(rect)


        # هدف
        ax.add_patch(plt.Circle((self.goal_xy[0], self.goal_xy[1]), self.cfg.goal_radius, color='green', alpha=0.5))


        # ربات
        x, y, th, v, w = self.model.state
        ax.add_patch(plt.Circle((x, y), self.cfg.robot_radius, color='blue', alpha=0.9))
        ax.arrow(x, y, 0.5*np.cos(th), 0.5*np.sin(th), head_width=0.15, length_includes_head=True)


        # مسیر طی‌شده
        if len(self.trajectory) >= 2:
            traj = np.vstack(self.trajectory)
            ax.plot(traj[:,0], traj[:,1], lw=2)


        self.render_fig.canvas.draw()
        self.render_fig.canvas.flush_events()
        time.sleep(self.cfg.dt)


    def close(self):
        if self.render_fig is not None:
            plt.close(self.render_fig)
            self.render_fig = None
            self.render_ax = None




# میان‌بری برای ساخت محیط از config.py
def make_env_from_config(cfg_module):
    cfg = WorldConfig(
        width=cfg_module.WORLD_WIDTH,
        height=cfg_module.WORLD_HEIGHT,
        robot_radius=cfg_module.ROBOT_RADIUS,
        lidar_n=cfg_module.LIDAR_N_RAYS,
        lidar_fov_deg=cfg_module.LIDAR_FOV_DEG,
        lidar_max_range=cfg_module.LIDAR_MAX_RANGE,
        lidar_noise_std=cfg_module.LIDAR_NOISE_STD,
        dt=cfg_module.DT,
        v_limits=(cfg_module.V_MIN, cfg_module.V_MAX),
        w_limits=(cfg_module.W_MIN, cfg_module.W_MAX),
        a_limits=(cfg_module.A_V_MAX, cfg_module.A_W_MAX),
        goal_radius=cfg_module.GOAL_RADIUS,
        safe_distance=cfg_module.SAFE_DISTANCE,
        max_steps=cfg_module.MAX_STEPS,
        n_circles=cfg_module.N_CIRCLES,
        n_boxes=cfg_module.N_BOXES,
        min_gap=cfg_module.MIN_GAP,
        obstacle_tries=cfg_module.OBSTACLE_TRIES,
        seed=cfg_module.N_SEED
    )


    reward_cfg = dict(
        R_COLLISION=cfg_module.R_COLLISION,
        R_GOAL=cfg_module.R_GOAL,
        K_PROGRESS=cfg_module.K_PROGRESS,
        K_HEADING=cfg_module.K_HEADING,
        K_TIME=cfg_module.K_TIME,
        K_TURNING=cfg_module.K_TURNING,
        K_SMOOTH_A=cfg_module.K_SMOOTH_A,
        K_JERK=cfg_module.K_JERK,
        K_CLEARANCE=cfg_module.K_CLEARANCE,
        K_BOUNDARY=cfg_module.K_BOUNDARY,
    )
    return MobileRobotNavEnv(cfg, reward_cfg)
