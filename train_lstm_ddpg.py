# # -*- coding: utf-8 -*-
# import os
# import argparse
# import numpy as np
# import torch
# from time import time

# import config as cfg
# from envs.mobile_robot_env import make_env_from_config

# from rddpg_lstm import RDDPG, SeqReplay

# def make_env_fn(seed=None):
#     def _init():
#         env = make_env_from_config(cfg)
#         if seed is not None:
#             env.reset(seed=seed)
#         return env
#     return _init

# def main(args):
#     logdir = args.logdir
#     os.makedirs(logdir, exist_ok=True)

#     # یک محیط تکی برای توالی‌ها (n_envs=1)
#     env = make_env_fn(seed=cfg.N_SEED)()
#     obs, _ = env.reset()

#     s_dim = env.observation_space.shape[0]
#     a_dim = env.action_space.shape[0]
#     a_high = float(env.action_space.high[0])

#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     agent = RDDPG(
#         state_dim=s_dim, action_dim=a_dim, action_high=a_high,
#         actor_lr=cfg.LR, critic_lr=max(cfg.LR * 10, 1e-3),
#         gamma=cfg.GAMMA, tau=cfg.TAU,
#         rnn_hidden=getattr(cfg, "RNN_HIDDEN", 128),
#         rnn_layers=getattr(cfg, "RNN_LAYERS", 1),
#         fc_hidden=getattr(cfg, "FC_HIDDEN", 256),
#         device=device
#     )

#     rb = SeqReplay(
#         capacity_eps=getattr(cfg, "REPLAY_EPISODES", 1000),
#         seq_len=getattr(cfg, "SEQ_LEN", 16),
#         device=device
#     )

#     total_steps = cfg.TOTAL_TIMESTEPS
#     start_steps = getattr(cfg, "START_STEPS", 5000)
#     batch_size  = cfg.BATCH_SIZE
#     noise_std   = getattr(cfg, "SIGMA_NOISE_A", 0.1)
#     eval_freq   = getattr(cfg, "EVAL_FREQ", 10000)
#     n_eval_ep   = getattr(cfg, "EVAL_EPISODES", 5)

#     rb.start_episode()
#     h_actor = None
#     best_eval = -1e9
#     t0 = time()

#     def evaluate(n_episodes=n_eval_ep):
#         returns = []
#         for _ in range(n_episodes):
#             s, _ = env.reset(seed=cfg.N_SEED + 123)
#             h = None
#             done = False
#             ep_ret = 0.0
#             while not done:
#                 a, h = agent.act(s, h, noise_std=0.0)
#                 s2, r, d, tr, _ = env.step(a)
#                 done = bool(d or tr)
#                 ep_ret += float(r)
#                 s = s2
#             returns.append(ep_ret)
#         return float(np.mean(returns))

#     for t in range(1, total_steps + 1):
#         a, h_actor = agent.act(obs, h_actor, noise_std=noise_std)
#         obs2, r, d, tr, _ = env.step(a)
#         done = bool(d or tr)
#         rb.add(obs, a, float(r), obs2, float(done))
#         obs = obs2

#         if done:
#             obs, _ = env.reset()
#             rb.start_episode()
#             h_actor = None

#         if t >= start_steps and len(rb.buf) >= 5:
#             batch = rb.sample(batch_size)
#             logs = agent.train_step(batch)

#         if t % max(1000, eval_freq) == 0:
#             avg_ret = evaluate()
#             if avg_ret > best_eval:
#                 best_eval = avg_ret
#                 torch.save(agent.actor.state_dict(), os.path.join(logdir, "best_actor.pt"))
#                 torch.save(agent.critic.state_dict(), os.path.join(logdir, "best_critic.pt"))
#             print(f"[RDDPG/LSTM] step={t} avg_return={avg_ret:.2f} best={best_eval:.2f} elapsed={time()-t0:.1f}s")

#     # save final
#     torch.save(agent.actor.state_dict(), os.path.join(logdir, "actor_final.pt"))
#     torch.save(agent.critic.state_dict(), os.path.join(logdir, "critic_final.pt"))
#     print(f"Training done. Models saved under: {logdir}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--logdir", type=str, default="logs_rddpg")
#     args = parser.parse_args()
#     main(args)

# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import torch
from time import time

import config as cfg
from envs.mobile_robot_env import make_env_from_config

from rddpg_lstm import RDDPG, SeqReplay


def make_env_fn(seed=None):
    def _init():
        env = make_env_from_config(cfg)
        if seed is not None:
            env.reset(seed=seed)
        return env
    return _init


def main(args):
    logdir = args.logdir
    os.makedirs(logdir, exist_ok=True)

    # محیط
    env = make_env_fn(seed=cfg.N_SEED)()
    obs, _ = env.reset()

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_high = float(env.action_space.high[0])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] Using: {device}")

    # ساخت عامل با تنظیمات جدید شبکه
    agent = RDDPG(
        state_dim=s_dim,
        action_dim=a_dim,
        action_high=a_high,
        actor_lr=getattr(cfg, "LR_ACTOR", getattr(cfg, "LR", 1e-4)),
        critic_lr=getattr(cfg, "LR_CRITIC", 1e-3),
        gamma=cfg.GAMMA,
        tau=cfg.TAU,
        rnn_hidden=getattr(cfg, "RNN_HIDDEN", 128),
        rnn_layers=getattr(cfg, "RNN_LAYERS", 1),
        fc_hidden=getattr(cfg, "FC_HIDDEN", 256),
        policy_delay=getattr(cfg, "POLICY_DELAY", 2),
        target_noise=getattr(cfg, "TARGET_NOISE", 0.2),
        target_noise_clip=getattr(cfg, "TARGET_NOISE_CLIP", 0.5),
        burnin=getattr(cfg, "BURNIN", 4),
        n_step=getattr(cfg, "N_STEP", 1),
        device=device,
    )

    # بافر حافظه توالی‌ها
    rb = SeqReplay(
        capacity_eps=getattr(cfg, "REPLAY_EPISODES", 2000),
        seq_len=getattr(cfg, "SEQ_LEN", 32),
        device=device,
    )

    total_steps = cfg.TOTAL_TIMESTEPS
    start_steps = getattr(cfg, "START_STEPS", 10000)
    batch_size = cfg.BATCH_SIZE
    eval_freq = getattr(cfg, "EVAL_FREQ", 10000)
    n_eval_ep = getattr(cfg, "EVAL_EPISODES", 5)

    # نویز برای اکتشاف
    noise_init = getattr(cfg, "SIGMA_NOISE_A", 0.2)
    noise_final = getattr(cfg, "SIGMA_MIN", 0.05)

    def noise_schedule(t):
        frac = min(1.0, t / (total_steps * 0.5))
        return noise_init + (noise_final - noise_init) * frac

    rb.start_episode()
    h_actor = None
    best_eval = -1e9
    t0 = time()

    # تابع ارزیابی
    @torch.no_grad()
    def evaluate(n_episodes=n_eval_ep):
        returns = []
        for k in range(n_episodes):
            s, _ = env.reset(seed=cfg.N_SEED + 123 + k)
            h = None
            done = False
            ep_ret = 0.0
            steps = 0
            while not done and steps < getattr(cfg, "MAX_STEPS", 1000):
                a, h = agent.act(s, h, noise_std=0.0)
                s2, r, d, tr, _ = env.step(a)
                done = bool(d or tr)
                ep_ret += float(r)
                s = s2
                steps += 1
            returns.append(ep_ret)
        return float(np.mean(returns))

    # حلقه‌ی آموزش
    for t in range(1, total_steps + 1):
        a_noise = noise_schedule(t)
        a, h_actor = agent.act(obs, h_actor, noise_std=a_noise)
        obs2, r, d, tr, _ = env.step(a)
        done = bool(d or tr)
        rb.add(obs, a, float(r), obs2, float(done))
        obs = obs2

        # اگر اپیزود تموم شد
        if done:
            obs, _ = env.reset()
            rb.start_episode()
            h_actor = None

        # یادگیری
        if t >= start_steps and len(rb.buf) >= 10:
            batch = rb.sample(batch_size)
            logs = agent.train_step(batch)

        # ارزیابی
        if t % eval_freq == 0:
            avg_ret = evaluate()
            if avg_ret > best_eval:
                best_eval = avg_ret
                torch.save(agent.actor.state_dict(), os.path.join(logdir, "best_actor.pt"))
                torch.save(agent.critic1.state_dict(), os.path.join(logdir, "best_critic1.pt"))
                torch.save(agent.critic2.state_dict(), os.path.join(logdir, "best_critic2.pt"))
            print(f"[RDDPG/LSTM+] step={t} avg_return={avg_ret:.2f} best={best_eval:.2f} elapsed={time()-t0:.1f}s")

    # ذخیره مدل نهایی
    torch.save(agent.actor.state_dict(), os.path.join(logdir, "actor_final.pt"))
    torch.save(agent.critic1.state_dict(), os.path.join(logdir, "critic1_final.pt"))
    torch.save(agent.critic2.state_dict(), os.path.join(logdir, "critic2_final.pt"))

    print(f"✅ Training done. Models saved under: {logdir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="logs_rddpg_lstm")
    args = parser.parse_args()
    main(args)
