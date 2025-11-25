# # -*- coding: utf-8 -*-
# import os
# import math
# import time
# import argparse
# import numpy as np
# from dataclasses import dataclass
# from typing import List, Tuple, Deque
# from collections import deque, namedtuple

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # ====== محیط و کانفیگ شما ======
# import config as cfg
# from envs.mobile_robot_env import make_env_from_config

# # ====== فقط برای حالت DDPG معمولی (SB3) ======
# from stable_baselines3 import DDPG
# from stable_baselines3.common.noise import NormalActionNoise
# from stable_baselines3.common.callbacks import EvalCallback
# from stable_baselines3.common.env_util import make_vec_env


# # =====================================================================================
# #                                   ابزارهای عمومی
# # =====================================================================================
# def make_env_fn(render_mode="none", seed=None):
#     def _init():
#         env = make_env_from_config(cfg)
#         if render_mode == "human":
#             env.metadata["render_modes"] = ["human"]
#         if seed is not None:
#             env.reset(seed=seed)
#         return env
#     return _init


# # =====================================================================================
# #                      پیاده‌سازی LSTM-DDPG (Actor + Critic دنباله‌ای)
# #                 (این بخش فقط وقتی --algo lstm-ddpg انتخاب شود استفاده می‌شود)
# # =====================================================================================

# # ---- نویز OU برای اکتشاف پیوسته ----
# class OUNoise:
#     def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
#         self.mu = mu; self.theta = theta; self.sigma = sigma
#         self.state = np.ones(size, dtype=np.float32) * self.mu
#     def reset(self):
#         self.state[:] = self.mu
#     def sample(self):
#         dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(*self.state.shape)
#         self.state = self.state + dx
#         return self.state

# # ---- کمک‌کننده اولیه‌سازی ----
# def fanin_init(tensor, fanin=None):
#     fanin = fanin or tensor.size(0)
#     bound = 1.0 / math.sqrt(fanin)
#     with torch.no_grad():
#         tensor.uniform_(-bound, bound)

# # ---- شبکه Actor با LSTM ----
# class ActorLSTM(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden=256, lstm_layers=1, act_limit=1.0):
#         super().__init__()
#         self.enc = nn.Linear(state_dim, hidden)
#         self.lstm = nn.LSTM(hidden, hidden, lstm_layers, batch_first=True)
#         self.h1 = nn.Linear(hidden, hidden)
#         self.h2 = nn.Linear(hidden, action_dim)
#         self.act_limit = act_limit
#         self.reset_parameters()

#     def reset_parameters(self):
#         fanin_init(self.enc.weight)
#         nn.init.zeros_(self.enc.bias)
#         for name, p in self.lstm.named_parameters():
#             if 'weight' in name:
#                 nn.init.xavier_uniform_(p)
#             elif 'bias' in name:
#                 nn.init.zeros_(p)
#         fanin_init(self.h1.weight); nn.init.zeros_(self.h1.bias)
#         nn.init.uniform_(self.h2.weight, -3e-3, 3e-3)
#         nn.init.uniform_(self.h2.bias,   -3e-3, 3e-3)

#     def forward(self, x, h=None, mask=None):
#         # x: [B, T, state_dim]
#         z = F.relu(self.enc(x))
#         z, h = self.lstm(z, h)  # [B,T,H]
#         z = F.relu(self.h1(z))
#         a = torch.tanh(self.h2(z)) * self.act_limit  # [-1,1] * limit
#         if mask is not None:
#             a = a * mask.unsqueeze(-1)
#         return a, h

# # ---- شبکه Critic با LSTM ----
# class CriticLSTM(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden=256, lstm_layers=1):
#         super().__init__()
#         self.es = nn.Linear(state_dim, hidden)
#         self.ea = nn.Linear(action_dim, hidden)
#         self.lstm = nn.LSTM(2*hidden, hidden, lstm_layers, batch_first=True)
#         self.h1 = nn.Linear(hidden, hidden)
#         self.h2 = nn.Linear(hidden, 1)
#         self.reset_parameters()

#     def reset_parameters(self):
#         fanin_init(self.es.weight); nn.init.zeros_(self.es.bias)
#         fanin_init(self.ea.weight); nn.init.zeros_(self.ea.bias)
#         for name, p in self.lstm.named_parameters():
#             if 'weight' in name:
#                 nn.init.xavier_uniform_(p)
#             elif 'bias' in name:
#                 nn.init.zeros_(p)
#         fanin_init(self.h1.weight); nn.init.zeros_(self.h1.bias)
#         nn.init.uniform_(self.h2.weight, -3e-3, 3e-3)
#         nn.init.uniform_(self.h2.bias,   -3e-3, 3e-3)

#     def forward(self, s, a, h=None, mask=None):
#         # s: [B,T,sd], a: [B,T,ad]
#         z = torch.cat([F.relu(self.es(s)), F.relu(self.ea(a))], dim=-1)
#         z, h = self.lstm(z, h)  # [B,T,H]
#         z = F.relu(self.h1(z))
#         q = self.h2(z).squeeze(-1)  # [B,T]
#         if mask is not None:
#             q = q * mask
#         return q, h

# # ---- ساختار حافظه اپیزودی برای آموزش سکانسی ----
# Step = namedtuple('Step', ['s','a','r','s2','d'])  # d: done (1.0/0.0)

# class EpisodeBuffer:
#     def __init__(self, max_episodes=1000, device="cpu"):
#         self.storage: Deque[List[Step]] = deque(maxlen=max_episodes)
#         self.device = device

#     def push_episode(self, steps: List[Step]):
#         self.storage.append(steps)

#     def __len__(self):
#         return len(self.storage)

#     def sample(self, batch_size, seq_len):
#         assert len(self.storage) > 0, "Replay buffer is empty."
#         batch_s, batch_a, batch_r, batch_s2, batch_d, masks = [],[],[],[],[],[]
#         for _ in range(batch_size):
#             ep = self.storage[np.random.randint(len(self.storage))]
#             if len(ep) <= 1:
#                 ep = ep + ep  # جلوگیری از طول صفر
#             start = np.random.randint(0, max(1, len(ep) - seq_len + 1))
#             seq = ep[start:start+seq_len]
#             L = len(seq)
#             if L < seq_len:
#                 pad = [ep[-1]] * (seq_len - L)
#                 seq = seq + pad
#             s  = np.stack([t.s for t in seq], 0)
#             a  = np.stack([t.a for t in seq], 0)
#             r  = np.stack([t.r for t in seq], 0)
#             s2 = np.stack([t.s2 for t in seq], 0)
#             d  = np.stack([t.d for t in seq], 0)
#             m  = np.zeros(seq_len, dtype=np.float32); m[:L] = 1.0
#             batch_s.append(s); batch_a.append(a); batch_r.append(r)
#             batch_s2.append(s2); batch_d.append(d); masks.append(m)

#         def to_t(x): return torch.as_tensor(np.stack(x,0), dtype=torch.float32, device=self.device)
#         return to_t(batch_s), to_t(batch_a), to_t(batch_r), to_t(batch_s2), to_t(batch_d), to_t(masks)

# # ---- عامل DDPG با LSTM ----
# class DDPG_LSTM:
#     def __init__(self, state_dim, action_dim, act_limit=1.0, device="cpu",
#                  gamma=0.99, tau=5e-3, lr=3e-4, hidden=256, lstm_layers=1,
#                  buffer_episodes=2000, batch_size=64, seq_len=32):
#         self.device = device
#         self.gamma = gamma
#         self.tau   = tau
#         self.act_limit = act_limit
#         self.batch_size = batch_size
#         self.seq_len = seq_len

#         self.actor = ActorLSTM(state_dim, action_dim, hidden, lstm_layers, act_limit).to(device)
#         self.actor_tgt = ActorLSTM(state_dim, action_dim, hidden, lstm_layers, act_limit).to(device)
#         self.critic = CriticLSTM(state_dim, action_dim, hidden, lstm_layers).to(device)
#         self.critic_tgt = CriticLSTM(state_dim, action_dim, hidden, lstm_layers).to(device)

#         self.actor_tgt.load_state_dict(self.actor.state_dict())
#         self.critic_tgt.load_state_dict(self.critic.state_dict())

#         self.pi_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
#         self.q_opt  = torch.optim.Adam(self.critic.parameters(), lr=lr)

#         self.noise = OUNoise(action_dim, sigma=cfg.SIGMA_NOISE_A)  # از کانفیگ
#         self.buffer = EpisodeBuffer(max_episodes=buffer_episodes, device=device)

#     @torch.no_grad()
#     def act(self, s, h=None, noise=True):
#         # s: np.array [state_dim]
#         s_t = torch.as_tensor(s, dtype=torch.float32, device=self.device).view(1,1,-1)
#         a, h = self.actor(s_t, h)
#         a = a.view(-1).cpu().numpy()
#         if noise:
#             a = np.clip(a + self.noise.sample(), -self.act_limit, self.act_limit)
#         return a, h

#     def soft_update(self, net, tgt):
#         for p, p_t in zip(net.parameters(), tgt.parameters()):
#             p_t.data.mul_(1.0 - self.tau)
#             p_t.data.add_(self.tau * p.data)

#     def train_step(self):
#         s, a, r, s2, d, m = self.buffer.sample(self.batch_size, self.seq_len)
#         with torch.no_grad():
#             a2, _ = self.actor_tgt(s2)                  # [B,T,A]
#             q2, _ = self.critic_tgt(s2, a2, mask=m)     # [B,T]
#             y = r.squeeze(-1) + (1.0 - d.squeeze(-1)) * self.gamma * q2
#             y = y * m

#         q, _ = self.critic(s, a, mask=m)
#         q_loss = ((q - y)**2 * m).sum() / (m.sum() + 1e-6)
#         self.q_opt.zero_grad(set_to_none=True)
#         q_loss.backward()
#         nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
#         self.q_opt.step()

#         a_pi, _ = self.actor(s, mask=m)
#         q_pi, _ = self.critic(s, a_pi, mask=m)
#         pi_loss = -(q_pi * m).sum() / (m.sum() + 1e-6)
#         self.pi_opt.zero_grad(set_to_none=True)
#         pi_loss.backward()
#         nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
#         self.pi_opt.step()

#         self.soft_update(self.actor, self.actor_tgt)
#         self.soft_update(self.critic, self.critic_tgt)
#         return float(q_loss.item()), float(pi_loss.item())

#     def save(self, path_dir):
#         os.makedirs(path_dir, exist_ok=True)
#         torch.save({
#             "actor": self.actor.state_dict(),
#             "critic": self.critic.state_dict(),
#             "actor_tgt": self.actor_tgt.state_dict(),
#             "critic_tgt": self.critic_tgt.state_dict(),
#             "cfg": {
#                 "gamma": self.gamma, "tau": self.tau,
#                 "seq_len": self.seq_len, "batch": self.batch_size,
#             }
#         }, os.path.join(path_dir, "lstm_ddpg_final.pt"))

# # =====================================================================================
# #                         حلقه آموزش یکپارچه (دو حالت قابل انتخاب)
# # =====================================================================================
# def train_sb3_ddpg(args):
#     logdir = args.logdir
#     os.makedirs(logdir, exist_ok=True)

#     # محیط‌های برداری برای آموزش و ارزیابی
#     train_env = make_vec_env(make_env_fn(seed=cfg.N_SEED), n_envs=args.n_envs, seed=cfg.N_SEED)
#     eval_env  = make_vec_env(make_env_fn(seed=cfg.N_SEED+1), n_envs=1, seed=cfg.N_SEED+1)

#     # نویز اکشن
#     action_noise = NormalActionNoise(mean=np.zeros(2), sigma=np.array([cfg.SIGMA_NOISE_A, cfg.SIGMA_NOISE_A], dtype=np.float32))

#     policy_kwargs = dict(net_arch=dict(pi=[256, 256], qf=[256, 256]), activation_fn=torch.nn.ReLU)

#     model = DDPG(
#         "MlpPolicy",
#         train_env,
#         learning_rate=cfg.LR,
#         buffer_size=cfg.BUFFER_SIZE,
#         batch_size=cfg.BATCH_SIZE,
#         gamma=cfg.GAMMA,
#         tau=cfg.TAU,
#         tensorboard_log=logdir,
#         action_noise=action_noise,
#         verbose=1,
#         policy_kwargs=policy_kwargs,
#         train_freq=(1, "step"),
#         gradient_steps=1,
#         device="auto"
#     )

#     eval_cb = EvalCallback(
#         eval_env,
#         best_model_save_path=logdir,
#         log_path=logdir,
#         eval_freq=cfg.EVAL_FREQ // max(1, args.n_envs),
#         n_eval_episodes=cfg.EVAL_EPISODES,
#         deterministic=False,
#         render=False
#     )

#     model.learn(total_timesteps=cfg.TOTAL_TIMESTEPS, callback=eval_cb, progress_bar=True)
#     model.save(os.path.join(logdir, "ddpg_final"))
#     print(f"[SB3-DDPG] Training done. Models saved under: {logdir}")


# def evaluate_env(env, agent_act_fn, episodes=5, render=False):
#     scores = []
#     for _ in range(episodes):
#         obs, _ = env.reset()
#         done = False; trunc = False
#         ep_ret = 0.0
#         h = None
#         while not (done or trunc):
#             a, h = agent_act_fn(obs, h)
#             obs, r, done, trunc, info = env.step(a)
#             ep_ret += r
#             if render and hasattr(env, "render"):
#                 env.render()
#         scores.append(ep_ret)
#     return float(np.mean(scores)), float(np.std(scores))


# def train_lstm_ddpg(args):
#     logdir = args.logdir
#     os.makedirs(logdir, exist_ok=True)

#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     # محیط‌های آموزش و ارزیابی (تکی برای مدیریت LSTM state ساده‌تر)
#     env = make_env_fn(seed=cfg.N_SEED)()
#     eval_env = make_env_fn(seed=cfg.N_SEED + 1)()

#     obs_dim = env.observation_space.shape[0]
#     act_dim = env.action_space.shape[0]
#     act_limit = float(env.action_space.high[0])  # باید 1.0 باشد طبق محیط شما

#     agent = DDPG_LSTM(
#         state_dim=obs_dim,
#         action_dim=act_dim,
#         act_limit=act_limit,
#         device=device,
#         gamma=cfg.GAMMA,
#         tau=cfg.TAU,
#         lr=cfg.LR,
#         hidden=256,
#         lstm_layers=1,
#         buffer_episodes=2000,
#         batch_size=cfg.BATCH_SIZE,
#         seq_len=32,
#     )

#     total_timesteps = cfg.TOTAL_TIMESTEPS
#     eval_freq = max(1, cfg.EVAL_FREQ)
#     n_eval_eps = cfg.EVAL_EPISODES

#     t_steps = 0
#     best_eval = -1e9
#     ep_idx = 0

#     while t_steps < total_timesteps:
#         obs, _ = env.reset()
#         done = False; trunc = False
#         h = None
#         ep_steps = 0
#         episode: List[Step] = []

#         # اکتشاف نویزی در طول اپیزود
#         agent.noise.reset()

#         while not (done or trunc):
#             a, h = agent.act(obs, h, noise=True)
#             next_obs, r, done, trunc, info = env.step(a)
#             d_flag = 1.0 if (done or trunc) else 0.0
#             episode.append(Step(obs.astype(np.float32),
#                                 a.astype(np.float32),
#                                 np.array([r], dtype=np.float32),
#                                 next_obs.astype(np.float32),
#                                 np.array([d_flag], dtype=np.float32)))
#             obs = next_obs
#             t_steps += 1
#             ep_steps += 1

#             # ارزیابی دوره‌ای
#             if t_steps % eval_freq == 0:
#                 mean_r, std_r = evaluate_env(eval_env,
#                                              agent_act_fn=lambda o, h_: agent.act(o, h_, noise=False),
#                                              episodes=n_eval_eps,
#                                              render=False)
#                 print(f"[LSTM-DDPG] Steps {t_steps}/{total_timesteps} | EvalReturn: {mean_r:.2f} ± {std_r:.2f}")
#                 # ذخیره بهترین
#                 if mean_r > best_eval:
#                     best_eval = mean_r
#                     agent.save(logdir)
#                     # یک فایل نشانگر بهترین
#                     with open(os.path.join(logdir, "best_eval.txt"), "w", encoding="utf-8") as f:
#                         f.write(f"best_mean_return={mean_r:.4f}\nsteps={t_steps}\n")

#             # چند آپدیت سبک بین راه (اگر دیتابیس به حداقل برسد)
#             if len(agent.buffer) > 5:
#                 for _ in range(1):
#                     ql, pl = agent.train_step()

#             if t_steps >= total_timesteps:
#                 break

#         # پایان اپیزود: اضافه به بافر و آموزش بیشتر
#         agent.buffer.push_episode(episode)
#         ep_idx += 1

#         # چند آپدیت اضافه به‌ازای هر اپیزود (تناسب با طول اپیزود)
#         if len(agent.buffer) > 5:
#             updates = max(1, ep_steps // 4)
#             for _ in range(updates):
#                 agent.train_step()

#     # ذخیره نهایی
#     agent.save(logdir)
#     print(f"[LSTM-DDPG] Training done. Models saved under: {logdir}")


# # =====================================================================================
# #                                          Main
# # =====================================================================================
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--algo", type=str, default="ddpg", choices=["ddpg", "lstm-ddpg"],
#                         help="ddpg (SB3) or lstm-ddpg (custom recurrent)")
#     parser.add_argument("--logdir", type=str, default="logs_compare",
#                         help="پوشه لاگ/مدل‌ها. برای هر حالت، خودت اسم مناسب بده.")
#     parser.add_argument("--n_envs", type=int, default=1, help="فقط برای حالت SB3-DDPG کاربرد دارد.")
#     args = parser.parse_args()

#     # ایجاد پوشه خروجی با پسوند نوع الگوریتم (برای تفکیک نتایج)
#     algo_tag = "sb3_ddpg" if args.algo == "ddpg" else "lstm_ddpg"
#     if os.path.basename(args.logdir) == args.logdir:
#         # اگر کاربر فقط نام ساده داد، پسوند بزنیم
#         args.logdir = f"{args.logdir}_{algo_tag}"
#     else:
#         # اگر مسیر کامل داد، همان را استفاده می‌کنیم
#         pass

#     if args.algo == "ddpg":
#         train_sb3_ddpg(args)
#     else:
#         train_lstm_ddpg(args)

# #python train_ddpg_compare.py --algo ddpg
# #python train_ddpg_compare.py --algo lstm-ddpg

# -*- coding: utf-8 -*-
"""
train_ddpg_compare.py
نسخه‌ی کامل DDPG و LSTM-DDPG با نمایش آنلاین ریواردها
"""
import os
import math
import time
import argparse
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Deque
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ====== محیط و کانفیگ ======
import config as cfg
from envs.mobile_robot_env import make_env_from_config

# ====== برای حالت DDPG معمولی (SB3) ======
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env


# =====================================================================================
#                                   ابزارهای عمومی
# =====================================================================================
def make_env_fn(render_mode="none", seed=None):
    def _init():
        env = make_env_from_config(cfg)
        if render_mode == "human":
            env.metadata["render_modes"] = ["human"]
        if seed is not None:
            env.reset(seed=seed)
        return env
    return _init


# =====================================================================================
#                      پیاده‌سازی LSTM-DDPG (Actor + Critic دنباله‌ای)
# =====================================================================================

class OUNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu; self.theta = theta; self.sigma = sigma
        self.state = np.ones(size, dtype=np.float32) * self.mu
    def reset(self): self.state[:] = self.mu
    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(*self.state.shape)
        self.state += dx
        return self.state

def fanin_init(tensor, fanin=None):
    fanin = fanin or tensor.size(0)
    bound = 1.0 / math.sqrt(fanin)
    with torch.no_grad():
        tensor.uniform_(-bound, bound)

class ActorLSTM(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256, lstm_layers=1, act_limit=1.0):
        super().__init__()
        self.enc = nn.Linear(state_dim, hidden)
        self.lstm = nn.LSTM(hidden, hidden, lstm_layers, batch_first=True)
        self.h1 = nn.Linear(hidden, hidden)
        self.h2 = nn.Linear(hidden, action_dim)
        self.act_limit = act_limit
        self.reset_parameters()

    def reset_parameters(self):
        fanin_init(self.enc.weight); nn.init.zeros_(self.enc.bias)
        for name, p in self.lstm.named_parameters():
            if 'weight' in name: nn.init.xavier_uniform_(p)
            elif 'bias' in name: nn.init.zeros_(p)
        fanin_init(self.h1.weight); nn.init.zeros_(self.h1.bias)
        nn.init.uniform_(self.h2.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.h2.bias,   -3e-3, 3e-3)

    def forward(self, x, h=None, mask=None):
        z = F.relu(self.enc(x))
        z, h = self.lstm(z, h)
        z = F.relu(self.h1(z))
        a = torch.tanh(self.h2(z)) * self.act_limit
        if mask is not None:
            a = a * mask.unsqueeze(-1)
        return a, h

class CriticLSTM(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256, lstm_layers=1):
        super().__init__()
        self.es = nn.Linear(state_dim, hidden)
        self.ea = nn.Linear(action_dim, hidden)
        self.lstm = nn.LSTM(2*hidden, hidden, lstm_layers, batch_first=True)
        self.h1 = nn.Linear(hidden, hidden)
        self.h2 = nn.Linear(hidden, 1)
        self.reset_parameters()

    def reset_parameters(self):
        fanin_init(self.es.weight); nn.init.zeros_(self.es.bias)
        fanin_init(self.ea.weight); nn.init.zeros_(self.ea.bias)
        for name, p in self.lstm.named_parameters():
            if 'weight' in name: nn.init.xavier_uniform_(p)
            elif 'bias' in name: nn.init.zeros_(p)
        fanin_init(self.h1.weight); nn.init.zeros_(self.h1.bias)
        nn.init.uniform_(self.h2.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.h2.bias,   -3e-3, 3e-3)

    def forward(self, s, a, h=None, mask=None):
        z = torch.cat([F.relu(self.es(s)), F.relu(self.ea(a))], dim=-1)
        z, h = self.lstm(z, h)
        z = F.relu(self.h1(z))
        q = self.h2(z).squeeze(-1)
        if mask is not None:
            q = q * mask
        return q, h


Step = namedtuple('Step', ['s','a','r','s2','d'])

class EpisodeBuffer:
    def __init__(self, max_episodes=1000, device="cpu"):
        self.storage: Deque[List[Step]] = deque(maxlen=max_episodes)
        self.device = device
    def push_episode(self, steps: List[Step]): self.storage.append(steps)
    def __len__(self): return len(self.storage)

    def sample(self, batch_size, seq_len):
        assert len(self.storage) > 0
        batch_s, batch_a, batch_r, batch_s2, batch_d, masks = [],[],[],[],[],[]
        for _ in range(batch_size):
            ep = self.storage[np.random.randint(len(self.storage))]
            start = np.random.randint(0, max(1, len(ep)-seq_len+1))
            seq = ep[start:start+seq_len]
            L = len(seq)
            if L < seq_len:
                seq += [ep[-1]]*(seq_len-L)
            s  = np.stack([t.s for t in seq],0)
            a  = np.stack([t.a for t in seq],0)
            r  = np.stack([t.r for t in seq],0)
            s2 = np.stack([t.s2 for t in seq],0)
            d  = np.stack([t.d for t in seq],0)
            m  = np.zeros(seq_len,np.float32); m[:L]=1.0
            batch_s.append(s); batch_a.append(a); batch_r.append(r)
            batch_s2.append(s2); batch_d.append(d); masks.append(m)
        def to_t(x): return torch.as_tensor(np.stack(x,0),dtype=torch.float32,device=self.device)
        return to_t(batch_s),to_t(batch_a),to_t(batch_r),to_t(batch_s2),to_t(batch_d),to_t(masks)


class DDPG_LSTM:
    def __init__(self, state_dim, action_dim, act_limit=1.0, device="cpu",
                 gamma=0.99, tau=5e-3, lr=3e-4, hidden=256, lstm_layers=1,
                 buffer_episodes=2000, batch_size=64, seq_len=32):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.act_limit = act_limit
        self.batch_size = batch_size
        self.seq_len = seq_len

        self.actor = ActorLSTM(state_dim, action_dim, hidden, lstm_layers, act_limit).to(device)
        self.actor_tgt = ActorLSTM(state_dim, action_dim, hidden, lstm_layers, act_limit).to(device)
        self.critic = CriticLSTM(state_dim, action_dim, hidden, lstm_layers).to(device)
        self.critic_tgt = CriticLSTM(state_dim, action_dim, hidden, lstm_layers).to(device)

        self.actor_tgt.load_state_dict(self.actor.state_dict())
        self.critic_tgt.load_state_dict(self.critic.state_dict())

        self.pi_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.noise = OUNoise(action_dim, sigma=cfg.SIGMA_NOISE_A)
        self.buffer = EpisodeBuffer(max_episodes=buffer_episodes, device=device)

    @torch.no_grad()
    def act(self, s, h=None, noise=True):
        s_t = torch.as_tensor(s, dtype=torch.float32, device=self.device).view(1,1,-1)
        a, h = self.actor(s_t, h)
        a = a.view(-1).cpu().numpy()
        if noise:
            a = np.clip(a + self.noise.sample(), -self.act_limit, self.act_limit)
        return a, h

    def soft_update(self, net, tgt):
        for p, p_t in zip(net.parameters(), tgt.parameters()):
            p_t.data.mul_(1.0 - self.tau)
            p_t.data.add_(self.tau * p.data)

    def train_step(self):
        s,a,r,s2,d,m = self.buffer.sample(self.batch_size,self.seq_len)
        with torch.no_grad():
            a2,_=self.actor_tgt(s2)
            q2,_=self.critic_tgt(s2,a2,mask=m)
            y=r.squeeze(-1)+(1.0-d.squeeze(-1))*self.gamma*q2
            y=y*m
        q,_=self.critic(s,a,mask=m)
        q_loss=((q-y)**2*m).sum()/(m.sum()+1e-6)
        self.q_opt.zero_grad(set_to_none=True)
        q_loss.backward(); nn.utils.clip_grad_norm_(self.critic.parameters(),1.0); self.q_opt.step()
        a_pi,_=self.actor(s,mask=m); q_pi,_=self.critic(s,a_pi,mask=m)
        pi_loss=-(q_pi*m).sum()/(m.sum()+1e-6)
        self.pi_opt.zero_grad(set_to_none=True)
        pi_loss.backward(); nn.utils.clip_grad_norm_(self.actor.parameters(),1.0); self.pi_opt.step()
        self.soft_update(self.actor,self.actor_tgt); self.soft_update(self.critic,self.critic_tgt)
        return float(q_loss.item()),float(pi_loss.item())

    def save(self,path_dir):
        os.makedirs(path_dir,exist_ok=True)
        torch.save({
            "actor":self.actor.state_dict(),
            "critic":self.critic.state_dict()
        },os.path.join(path_dir,"lstm_ddpg_final.pt"))


# =====================================================================================
#                                   حالت SB3-DDPG
# =====================================================================================
def train_sb3_ddpg(args):
    logdir=args.logdir; os.makedirs(logdir,exist_ok=True)
    train_env=make_vec_env(make_env_fn(seed=cfg.N_SEED),n_envs=args.n_envs,seed=cfg.N_SEED)
    eval_env=make_vec_env(make_env_fn(seed=cfg.N_SEED+1),n_envs=1,seed=cfg.N_SEED+1)
    action_noise=NormalActionNoise(mean=np.zeros(2),sigma=np.array([cfg.SIGMA_NOISE_A]*2,dtype=np.float32))
    policy_kwargs=dict(net_arch=dict(pi=[256,256],qf=[256,256]),activation_fn=torch.nn.ReLU)
    model=DDPG("MlpPolicy",train_env,learning_rate=cfg.LR,buffer_size=cfg.BUFFER_SIZE,
               batch_size=cfg.BATCH_SIZE,gamma=cfg.GAMMA,tau=cfg.TAU,tensorboard_log=logdir,
               action_noise=action_noise,verbose=1,policy_kwargs=policy_kwargs,
               train_freq=(1,"step"),gradient_steps=1,device="auto")
    eval_cb=EvalCallback(eval_env,best_model_save_path=logdir,log_path=logdir,
                         eval_freq=cfg.EVAL_FREQ//max(1,args.n_envs),
                         n_eval_episodes=cfg.EVAL_EPISODES,deterministic=False,render=False)
    model.learn(total_timesteps=cfg.TOTAL_TIMESTEPS,callback=eval_cb,progress_bar=True)
    model.save(os.path.join(logdir,"ddpg_final"))
    print(f"[SB3-DDPG] Training done. Models saved under: {logdir}")


# =====================================================================================
#                                   حالت LSTM-DDPG با نمایش آنلاین
# =====================================================================================
def train_lstm_ddpg(args):
    logdir=args.logdir; os.makedirs(logdir,exist_ok=True)
    device="cuda" if torch.cuda.is_available() else "cpu"
    env=make_env_fn(seed=cfg.N_SEED)(); eval_env=make_env_fn(seed=cfg.N_SEED+1)()
    obs_dim=env.observation_space.shape[0]; act_dim=env.action_space.shape[0]; act_limit=float(env.action_space.high[0])
    agent=DDPG_LSTM(obs_dim,act_dim,act_limit,device=device,gamma=cfg.GAMMA,tau=cfg.TAU,lr=cfg.LR,
                    hidden=256,lstm_layers=1,buffer_episodes=2000,batch_size=cfg.BATCH_SIZE,seq_len=32)

    total_timesteps=cfg.TOTAL_TIMESTEPS; eval_freq=max(1,cfg.EVAL_FREQ); n_eval_eps=cfg.EVAL_EPISODES
    t_steps=0; best_eval=-1e9; rewards_window=deque(maxlen=10)

    while t_steps<total_timesteps:
        obs,_=env.reset(); done=False; trunc=False; h=None; ep_reward=0; episode=[]
        agent.noise.reset()
        while not(done or trunc):
            a,h=agent.act(obs,h,noise=True)
            next_obs,r,done,trunc,info=env.step(a)
            d_flag=1.0 if(done or trunc)else 0.0
            episode.append(Step(obs.astype(np.float32),a.astype(np.float32),
                                np.array([r],dtype=np.float32),
                                next_obs.astype(np.float32),
                                np.array([d_flag],dtype=np.float32)))
            obs=next_obs; ep_reward+=r; t_steps+=1

            if len(agent.buffer)>5:
                ql,pl=agent.train_step()
            if t_steps%500==0:
                print(f"[LSTM-DDPG] step={t_steps} | last_ep_reward={ep_reward:.2f}")

            if t_steps%eval_freq==0:
                mean_r,std_r=evaluate_env(eval_env,lambda o,h_:agent.act(o,h_,noise=False),n_eval_eps)
                print(f"[EVAL] step={t_steps}/{total_timesteps} | avg={mean_r:.2f} ± {std_r:.2f}")
                if mean_r>best_eval:
                    best_eval=mean_r; agent.save(logdir)
                    with open(os.path.join(logdir,"best_eval.txt"),"w") as f:f.write(f"best={mean_r:.3f}\n")
            if t_steps>=total_timesteps: break
        agent.buffer.push_episode(episode)
        rewards_window.append(ep_reward)
        avg_last=np.mean(rewards_window)
        print(f"[LSTM-DDPG] Episode done: ep_reward={ep_reward:.2f} | avg_last10={avg_last:.2f}")

    agent.save(logdir)
    print(f"[LSTM-DDPG] ✅ Training done. Models saved under: {logdir}")


# =====================================================================================
#                                   ارزیابی سریع
# =====================================================================================
def evaluate_env(env,agent_act_fn,episodes=5,render=False):
    scores=[]
    for _ in range(episodes):
        obs,_=env.reset(); done=False; trunc=False; ep_ret=0; h=None
        while not(done or trunc):
            a,h=agent_act_fn(obs,h); obs,r,done,trunc,info=env.step(a); ep_ret+=r
            if render and hasattr(env,"render"): env.render()
        scores.append(ep_ret)
    return float(np.mean(scores)),float(np.std(scores))


# =====================================================================================
#                                          Main
# =====================================================================================
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--algo",type=str,default="lstm-ddpg",choices=["ddpg","lstm-ddpg"])
    parser.add_argument("--logdir",type=str,default="logs_compare")
    parser.add_argument("--n_envs",type=int,default=1)
    args=parser.parse_args()
    algo_tag="sb3_ddpg" if args.algo=="ddpg" else "lstm_ddpg"
    if os.path.basename(args.logdir)==args.logdir:
        args.logdir=f"{args.logdir}_{algo_tag}"

    if args.algo=="ddpg": train_sb3_ddpg(args)
    else: train_lstm_ddpg(args)
