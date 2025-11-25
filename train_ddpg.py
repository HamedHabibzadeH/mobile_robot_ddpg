# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import torch

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env

import config as cfg
from envs.mobile_robot_env import make_env_from_config

def make_env_fn(render_mode="none", seed=None):
    def _init():
        env = make_env_from_config(cfg)
        if render_mode == "human":
            env.metadata["render_modes"] = ["human"]
        if seed is not None:
            env.reset(seed=seed)
        return env
    return _init

def main(args):
    logdir = args.logdir
    os.makedirs(logdir, exist_ok=True)

    # محیط‌های برداری برای آموزش و ارزیابی
    train_env = make_vec_env(make_env_fn(seed=cfg.N_SEED), n_envs=args.n_envs, seed=cfg.N_SEED)
    eval_env  = make_vec_env(make_env_fn(seed=cfg.N_SEED+1), n_envs=1, seed=cfg.N_SEED+1)

    # نویز اکشن
    action_noise = NormalActionNoise(mean=np.zeros(2), sigma=np.array([cfg.SIGMA_NOISE_A, cfg.SIGMA_NOISE_A], dtype=np.float32))

    policy_kwargs = dict(net_arch=dict(pi=[256, 256], qf=[256, 256]), activation_fn=torch.nn.ReLU)

    model = DDPG(
        "MlpPolicy",
        train_env,
        learning_rate=cfg.LR,
        buffer_size=cfg.BUFFER_SIZE,
        batch_size=cfg.BATCH_SIZE,
        gamma=cfg.GAMMA,
        tau=cfg.TAU,
        tensorboard_log=logdir,
        action_noise=action_noise,
        verbose=1,
        policy_kwargs=policy_kwargs,
        train_freq=(1, "step"),
        gradient_steps=1,
        device="auto"
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=logdir,
        log_path=logdir,
        eval_freq=cfg.EVAL_FREQ // args.n_envs,
        n_eval_episodes=cfg.EVAL_EPISODES,
        deterministic=False,
        render=False
    )

    model.learn(total_timesteps=cfg.TOTAL_TIMESTEPS, callback=eval_cb, progress_bar=True)
    model.save(os.path.join(logdir, "ddpg_final"))

    print(f"Training done. Models saved under: {logdir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="logs_ddpg")
    parser.add_argument("--n_envs", type=int, default=1)
    args = parser.parse_args()
    main(args)
