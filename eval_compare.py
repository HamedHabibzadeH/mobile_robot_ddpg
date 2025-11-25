# -*- coding: utf-8 -*-
import os
import time
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG

import config as cfg
from envs.mobile_robot_env import make_env_from_config

# ---- در صورت استفاده از LSTM-DDPG ----
try:
    from custom_ddpg_lstm_policy import ActorLSTM
except ImportError:
    ActorLSTM = None


def plot_trajectory(traj, goal, title="Trajectory", save_path=None):
    """نمودار مسیر حرکت ربات"""
    traj = np.array(traj)
    plt.figure(figsize=(8, 5))
    plt.plot(traj[:, 0], traj[:, 1], 'b-', lw=2, label="Path")
    plt.plot(traj[0, 0], traj[0, 1], 'go', markersize=10, label="Start")
    plt.plot(goal[0], goal[1], 'r*', markersize=12, label="Goal")
    plt.xlim(0, cfg.WORLD_WIDTH)
    plt.ylim(0, cfg.WORLD_HEIGHT)
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show(block=False)
    plt.pause(2)
    plt.close()


def rollout_ddpg(model_path, episodes=5, deterministic=True, sleep=0.0):
    """ارزیابی مدل DDPG (SB3)"""
    env = make_env_from_config(cfg)
    model = DDPG.load(model_path)
    successes, rewards = 0, []

    for ep in range(episodes):
        obs, info = env.reset(seed=cfg.N_SEED + 100 + ep)
        ep_reward = 0.0
        traj = [env.model.state[:2].copy()]

        for t in range(cfg.MAX_STEPS):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            traj.append(env.model.state[:2].copy())
            if sleep > 0:
                env.render()
                time.sleep(sleep)
            if terminated or truncated:
                if info.get("is_success", False):
                    successes += 1
                break

        rewards.append(ep_reward)
        print(f"[DDPG | EP {ep+1}] reward={ep_reward:.2f} success={info.get('is_success', False)} steps={t+1}")
        plot_trajectory(traj, env.goal_xy, title=f"DDPG Episode {ep+1}")

    print("=" * 60)
    print(f"DDPG Success rate: {successes}/{episodes}")
    print(f"DDPG Avg reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print("=" * 60)
    env.close()


def rollout_lstm_ddpg(model_path, episodes=5, sleep=0.0, device="cpu"):
    """ارزیابی مدل LSTM-DDPG ذخیره‌شده (.pt)"""
    if ActorLSTM is None:
        raise ImportError("فایل custom_ddpg_lstm_policy.py یا ActorLSTM در دسترس نیست!")

    env = make_env_from_config(cfg)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # مدل بازیابی
    actor = ActorLSTM(obs_dim, act_dim).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()

    print(f"Loaded LSTM-DDPG actor from {model_path}")
    successes, rewards = 0, []

    for ep in range(episodes):
        obs, info = env.reset(seed=cfg.N_SEED + 100 + ep)
        obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        h, c = actor.init_hidden(batch_size=1)
        ep_reward = 0.0
        traj = [env.model.state[:2].copy()]

        for t in range(cfg.MAX_STEPS):
            with torch.no_grad():
                action, (h, c) = actor(obs, (h, c))
                action = action.cpu().numpy()[0]

            obs_, reward, terminated, truncated, info = env.step(action)
            obs = torch.tensor(obs_, dtype=torch.float32, device=device).unsqueeze(0)
            ep_reward += reward
            traj.append(env.model.state[:2].copy())

            if sleep > 0:
                env.render()
                time.sleep(sleep)
            if terminated or truncated:
                if info.get("is_success", False):
                    successes += 1
                break

        rewards.append(ep_reward)
        print(f"[LSTM-DDPG | EP {ep+1}] reward={ep_reward:.2f} success={info.get('is_success', False)} steps={t+1}")
        plot_trajectory(traj, env.goal_xy, title=f"LSTM-DDPG Episode {ep+1}")

    print("=" * 60)
    print(f"LSTM-DDPG Success rate: {successes}/{episodes}")
    print(f"LSTM-DDPG Avg reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print("=" * 60)
    env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--algo", type=str, choices=["ddpg", "lstm-ddpg"], default="ddpg",
                   help="نوع الگوریتم برای ارزیابی: ddpg یا lstm-ddpg")
    p.add_argument("--model", type=str, default=None, help="مسیر مدل ذخیره‌شده")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--sleep", type=float, default=0.0)
    args = p.parse_args()

    # مسیر پیش‌فرض مدل‌ها
    if args.algo == "ddpg" and args.model is None:
        args.model = "logs_compare_sb3_ddpg/best_model.zip"
    elif args.algo == "lstm-ddpg" and args.model is None:
        args.model = "logs_compare_lstm_ddpg/lstm_ddpg_final.pt"

    print(f"🔍 Evaluating {args.algo.upper()} model from: {args.model}")

    if args.algo == "ddpg":
        rollout_ddpg(args.model, episodes=args.episodes, deterministic=True, sleep=args.sleep)
    elif args.algo == "lstm-ddpg":
        rollout_lstm_ddpg(args.model, episodes=args.episodes, sleep=args.sleep)
