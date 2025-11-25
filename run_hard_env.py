# -*- coding: utf-8 -*-
import time
from stable_baselines3 import DDPG

import config as cfg
from envs.mobile_robot_env import make_env_from_config

def test_harder_env(model_path="logs_ddpg/best_model.zip", episodes=3, deterministic=True, sleep=0.02):
    # تغییر پارامترهای محیط به حالت سخت‌تر
    cfg.N_CIRCLES = 6
    cfg.N_BOXES = 6
    cfg.MIN_GAP = 0.3

    # ساخت محیط سخت‌تر
    env = make_env_from_config(cfg)

    # بارگذاری مدل آموزش‌دیده
    model = DDPG.load(model_path)

    for ep in range(episodes):
        obs, info = env.reset()
        ep_reward = 0
        for step in range(cfg.MAX_STEPS):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            env.render()
            if sleep > 0:
                time.sleep(sleep)
            if terminated or truncated:
                print(f"Episode {ep+1} ended. Success={info.get('is_success', False)}, Reward={ep_reward:.2f}")
                break

    env.close()

if __name__ == "__main__":
    test_harder_env("logs_ddpg/best_model.zip", episodes=5, deterministic=True, sleep=0.02)
