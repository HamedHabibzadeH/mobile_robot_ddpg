# -*- coding: utf-8 -*-
import time
import argparse
import numpy as np
import torch

import config as cfg
from envs.mobile_robot_env import make_env_from_config
from rddpg_lstm import RDDPG   # الگوریتم LSTM-DDPG خودت

def rollout(model_dir, episodes=5, deterministic=True, sleep=0.0):
    """
    اجرای چند اپیزود با مدل LSTM-DDPG آموزش‌دیده.
    model_dir باید پوشه‌ای باشد که فایل‌های actor_final.pt یا best_actor.pt در آن ذخیره شده‌اند.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ساخت محیط
    env = make_env_from_config(cfg)
    successes = 0

    # مشخصات فضای حالت و عمل
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_high = float(env.action_space.high[0])

    # مدل LSTM-DDPG با همان ساختار آموزش
    agent = RDDPG(
        state_dim=s_dim, action_dim=a_dim, action_high=a_high,
        actor_lr=cfg.LR, critic_lr=cfg.LR, gamma=cfg.GAMMA, tau=cfg.TAU,
        rnn_hidden=getattr(cfg, "RNN_HIDDEN", 128),
        rnn_layers=getattr(cfg, "RNN_LAYERS", 1),
        fc_hidden=getattr(cfg, "FC_HIDDEN", 256),
        device=device
    )

    # تلاش برای بارگذاری بهترین مدل، در غیر این صورت مدل نهایی
    best_actor_path = f"{model_dir}/best_actor.pt"
    final_actor_path = f"{model_dir}/actor_final.pt"

    model_path = best_actor_path if os.path.exists(best_actor_path) else final_actor_path
    agent.actor.load_state_dict(torch.load(model_path, map_location=device))
    agent.actor.eval()

    print(f"Loaded model from: {model_path}")

    # اجرای اپیزودها
    for ep in range(episodes):
        obs, info = env.reset(seed=cfg.N_SEED + 100 + ep)
        ep_reward = 0.0
        h_actor = None  # وضعیت پنهان LSTM

        for t in range(cfg.MAX_STEPS):
            # پیش‌بینی اکشن با LSTM
            noise_std = 0.0 if deterministic else getattr(cfg, "SIGMA_NOISE_A", 0.1)
            action, h_actor = agent.act(obs, h_actor, noise_std=noise_std)

            # قدم در محیط
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

            env.render()
            if sleep > 0:
                time.sleep(sleep)

            # پایان اپیزود
            if terminated or truncated:
                if info.get("is_success", False):
                    successes += 1
                break

        print(f"[EP {ep+1}] reward={ep_reward:.2f} success={info.get('is_success', False)} steps={t+1}")

    print(f"Success rate: {successes}/{episodes}")
    env.close()


if __name__ == "__main__":
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="logs_rddpg")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    rollout(args.model_dir, episodes=args.episodes,
            deterministic=args.deterministic, sleep=args.sleep)
