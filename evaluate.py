# -*- coding: utf-8 -*-
import time
import argparse
import numpy as np
from stable_baselines3 import DDPG

import config as cfg
from envs.mobile_robot_env import make_env_from_config

def rollout(model_path, episodes=5, deterministic=True, sleep=0.0):
    env = make_env_from_config(cfg)
    successes = 0
    model = DDPG.load(model_path)

    for ep in range(episodes):
        obs, info = env.reset(seed=cfg.N_SEED+100+ep)
        ep_reward = 0.0
        for t in range(cfg.MAX_STEPS):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            env.render()
            if sleep > 0:
                time.sleep(sleep)
            if terminated or truncated:
                if info.get("is_success", False):
                    successes += 1
                break
        print(f"[EP {ep+1}] reward={ep_reward:.2f} success={info.get('is_success', False)} steps={t+1}")

    print(f"Success rate: {successes}/{episodes}")
    env.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="logs_ddpg/best_model.zip")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--sleep", type=float, default=0.0)
    args = p.parse_args()

    rollout(args.model, episodes=args.episodes, deterministic=args.deterministic, sleep=args.sleep)
