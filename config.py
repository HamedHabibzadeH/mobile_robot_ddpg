# -*- coding: utf-8 -*-

N_SEED = 42

# ابعاد محیط (متر)
WORLD_WIDTH  = 12.0
WORLD_HEIGHT = 8.0

# ربات
ROBOT_RADIUS = 0.20

# دینامیک
DT                = 0.08        # s
V_MIN, V_MAX      = 0.0, 1.2    # m/s
W_MIN, W_MAX      = -2.5, 2.5   # rad/s
A_V_MAX, A_W_MAX  = 0.8, 3.0    # m/s^2 , rad/s^2

# LiDAR
LIDAR_FOV_DEG     = 270.0
LIDAR_N_RAYS      = 64
LIDAR_MAX_RANGE   = 6.0         # m
LIDAR_NOISE_STD   = 0.01        # نسبی (روی مقدار نرمال‌شده 0..1)

# اپیزود
MAX_STEPS         = 650
GOAL_RADIUS       = 0.35        # m
SAFE_DISTANCE     = 0.50        # m  (حاشیه‌ی امن از موانع)

# موانع
N_CIRCLES         = 6
N_BOXES           = 4
MIN_GAP           = 0.6         # فاصله‌ی حداقل میان موانع و مسیرها
OBSTACLE_TRIES    = 64

# ضرایب ریوارد
R_COLLISION       = -50.0
R_GOAL            = +200.0
K_PROGRESS        = +3.0
K_HEADING         = +0.6
K_TIME            = -0.01
K_TURNING         = -0.08            # جریمه‌ی نرخ چرخش |w|
K_SMOOTH_A        = -0.05            # جریمه‌ی |a_v| و |a_w|
K_JERK            = -0.03            # جریمه‌ی تغییرات سریع فرمان
K_CLEARANCE       = -0.6             # جریمه‌ی نزدیک شدن بیش از حد به موانع
K_BOUNDARY        = -0.25            # نزدیک شدن به دیواره‌ها

# آموزش
TOTAL_TIMESTEPS   = 50_000
EVAL_FREQ         = 1_000
EVAL_EPISODES     = 8
BUFFER_SIZE       = 200_000
BATCH_SIZE        = 64
GAMMA             = 0.995
TAU               = 0.005
LR                = 3e-4
SIGMA_NOISE_A     = 0.3               # نویز اکشن (استاندارد نرمال)
