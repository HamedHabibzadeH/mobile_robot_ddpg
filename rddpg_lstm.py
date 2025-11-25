# # rddpg_lstm.py
# import math
# import copy
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim

# # ---------- Models ----------
# def fanin_init(tensor):
#     fan_in = tensor.size(0)
#     bound = 1.0 / math.sqrt(fan_in)
#     with torch.no_grad():
#         tensor.uniform_(-bound, bound)

# class ActorLSTM(nn.Module):
#     def __init__(self, state_dim, action_dim, action_high=1.0,
#                  rnn_hidden=128, rnn_layers=1, fc_hidden=256):
#         super().__init__()
#         self.rnn = nn.LSTM(input_size=state_dim, hidden_size=rnn_hidden,
#                            num_layers=rnn_layers, batch_first=True)
#         self.ln = nn.LayerNorm(rnn_hidden)
#         self.fc = nn.Sequential(
#             nn.Linear(rnn_hidden, fc_hidden), nn.ReLU(inplace=True),
#             nn.Linear(fc_hidden, action_dim), nn.Tanh()
#         )
#         self.action_high = float(action_high)
#         self.apply(self._init)

#     def _init(self, m):
#         if isinstance(m, nn.Linear):
#             fanin_init(m.weight); nn.init.zeros_(m.bias)

#     def forward(self, s, h=None):
#         # s: [B,T,obs]
#         y, h = self.rnn(s, h)
#         y = self.ln(y)
#         a = self.fc(y) * self.action_high
#         return a, h

#     def act_last(self, s1, h=None):
#         # s1: [B,1,obs]
#         a_seq, h = self.forward(s1, h)
#         return a_seq[:, -1, :], h

# class CriticLSTM(nn.Module):
#     def __init__(self, state_dim, action_dim, rnn_hidden=128, rnn_layers=1, fc_hidden=256):
#         super().__init__()
#         self.rnn = nn.LSTM(input_size=state_dim + action_dim, hidden_size=rnn_hidden,
#                            num_layers=rnn_layers, batch_first=True)
#         self.ln = nn.LayerNorm(rnn_hidden)
#         self.fc = nn.Sequential(
#             nn.Linear(rnn_hidden, fc_hidden), nn.ReLU(inplace=True),
#             nn.Linear(fc_hidden, 1)
#         )
#         self.apply(self._init)

#     def _init(self, m):
#         if isinstance(m, nn.Linear):
#             fanin_init(m.weight); nn.init.zeros_(m.bias)

#     def forward(self, s, a, h=None):
#         # s,a: [B,T,*]
#         x = torch.cat([s, a], dim=-1)
#         y, h = self.rnn(x, h)
#         y = self.ln(y)
#         q = self.fc(y)  # [B,T,1]
#         return q, h

# # ---------- Recurrent Replay (fixed-length segments) ----------
# class EpisodeBuf:
#     def __init__(self):
#         self.s, self.a, self.r, self.s2, self.d = [], [], [], [], []
#     def add(self, s, a, r, s2, d):
#         self.s.append(s); self.a.append(a); self.r.append(r); self.s2.append(s2); self.d.append(d)

# class SeqReplay:
#     def __init__(self, capacity_eps=1000, seq_len=16, device="cpu"):
#         self.capacity_eps = capacity_eps
#         self.seq_len = seq_len
#         self.device = device
#         self.buf = []

#     def start_episode(self):
#         self.buf.append(EpisodeBuf())
#         if len(self.buf) > self.capacity_eps:
#             self.buf.pop(0)

#     def add(self, s, a, r, s2, d):
#         if not self.buf: self.start_episode()
#         self.buf[-1].add(s, a, r, s2, d)
#         if d: self.start_episode()

#     def sample(self, batch_size):
#         Ss, As, Rs, S2s, Ds, Ms = [], [], [], [], [], []
#         for _ in range(batch_size):
#             ep = np.random.choice(self.buf)
#             while len(ep.s) < self.seq_len:
#                 ep = np.random.choice(self.buf)
#             t0 = np.random.randint(0, len(ep.s) - self.seq_len + 1)
#             t1 = t0 + self.seq_len
#             s  = np.asarray(ep.s [t0:t1], np.float32)
#             a  = np.asarray(ep.a [t0:t1], np.float32)
#             r  = np.asarray(ep.r [t0:t1], np.float32)[..., None]
#             s2 = np.asarray(ep.s2[t0:t1], np.float32)
#             d  = np.asarray(ep.d [t0:t1], np.float32)[..., None]
#             m  = np.ones_like(r, np.float32)  # چون segment ثابت است
#             Ss.append(s); As.append(a); Rs.append(r); S2s.append(s2); Ds.append(d); Ms.append(m)
#         to_t = lambda xs: torch.tensor(np.stack(xs), device=self.device)
#         return to_t(Ss), to_t(As), to_t(Rs), to_t(S2s), to_t(Ds), to_t(Ms)

# # ---------- Agent ----------
# class RDDPG:
#     def __init__(self, state_dim, action_dim, action_high,
#                  actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=5e-3,
#                  rnn_hidden=128, rnn_layers=1, fc_hidden=256, device="cpu"):
#         self.device = device
#         self.gamma = gamma
#         self.tau = tau
#         self.action_high = action_high

#         self.actor = ActorLSTM(state_dim, action_dim, action_high, rnn_hidden, rnn_layers, fc_hidden).to(device)
#         self.actor_tgt = copy.deepcopy(self.actor).to(device)
#         self.critic = CriticLSTM(state_dim, action_dim, rnn_hidden, rnn_layers, fc_hidden).to(device)
#         self.critic_tgt = copy.deepcopy(self.critic).to(device)

#         self.pi_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
#         self.q_opt  = optim.Adam(self.critic.parameters(), lr=critic_lr)
#         self.mse = nn.MSELoss(reduction="none")

#     @torch.no_grad()
#     def act(self, s, h=None, noise_std=0.1):
#         if not torch.is_tensor(s):
#             s = torch.tensor(s, dtype=torch.float32, device=self.device)
#         s = s.unsqueeze(0).unsqueeze(1)  # [1,1,D]
#         a1, h = self.actor.act_last(s, h)  # [1,A]
#         a = a1.squeeze(0)
#         if noise_std > 0:
#             a = a + noise_std * torch.randn_like(a)
#         a = torch.clamp(a, -self.action_high, self.action_high)
#         return a.cpu().numpy(), h

#     def _soft_update(self, net, tgt):
#         with torch.no_grad():
#             for p, tp in zip(net.parameters(), tgt.parameters()):
#                 tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

#     def train_step(self, batch):
#         S, A, R, S2, D, M = batch  # [B,T,...]

#         # ----- Critic -----
#         with torch.no_grad():
#             a2, _ = self.actor_tgt(S2)          # [B,T,Adim]
#             q2, _ = self.critic_tgt(S2, a2)     # [B,T,1]
#             y = R + (1.0 - D) * (self.gamma * q2)

#         q, _ = self.critic(S, A)
#         critic_loss = (self.mse(q, y) * M).sum() / M.sum().clamp_min(1.0)
#         self.q_opt.zero_grad(set_to_none=True)
#         critic_loss.backward()
#         torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
#         self.q_opt.step()

#         # ----- Actor -----
#         a_pi, _ = self.actor(S)
#         q_pi, _ = self.critic(S, a_pi)
#         actor_loss = (-(q_pi) * M).sum() / M.sum().clamp_min(1.0)
#         self.pi_opt.zero_grad(set_to_none=True)
#         actor_loss.backward()
#         torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
#         self.pi_opt.step()

#         # targets
#         self._soft_update(self.actor, self.actor_tgt)
#         self._soft_update(self.critic, self.critic_tgt)

#         return {
#             "critic_loss": float(critic_loss.item()),
#             "actor_loss":  float(actor_loss.item()),
#             "q_mean":      float(q.mean().item()),
#         }


# rddpg_lstm.py
# Recurrent TD3-style (LSTM) with twin critics, target policy smoothing, policy delay,
# burn-in masking, optional n-step targets, and sequence replay.

import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# =========================
# Utils
# =========================
def fanin_init(tensor):
    fan_in = max(1, tensor.size(0))
    bound = 1.0 / math.sqrt(fan_in)
    with torch.no_grad():
        tensor.uniform_(-bound, bound)

class LayerNorm1D(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(dim, eps=eps)
    def forward(self, x):  # [B,T,D]
        return self.ln(x)

# =========================
# Models
# =========================
class ActorLSTM(nn.Module):
    def __init__(self, state_dim, action_dim, action_high=1.0,
                 rnn_hidden=128, rnn_layers=1, fc_hidden=256):
        super().__init__()
        self.rnn = nn.LSTM(input_size=state_dim, hidden_size=rnn_hidden,
                           num_layers=rnn_layers, batch_first=True)
        self.ln = LayerNorm1D(rnn_hidden)
        self.fc = nn.Sequential(
            nn.Linear(rnn_hidden, fc_hidden), nn.ReLU(inplace=True),
            nn.Linear(fc_hidden, action_dim),
            nn.Tanh()
        )
        self.action_high = float(action_high)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            fanin_init(m.weight); nn.init.zeros_(m.bias)

    def forward(self, s, h=None):
        y, h = self.rnn(s, h)       # [B,T,H]
        y = self.ln(y)
        a = self.fc(y) * self.action_high
        return a, h

    def act_last(self, s1, h=None):
        # s1: [B,1,D]
        a_seq, h = self.forward(s1, h)
        return a_seq[:, -1, :], h

class CriticLSTM(nn.Module):
    def __init__(self, state_dim, action_dim,
                 rnn_hidden=128, rnn_layers=1, fc_hidden=256):
        super().__init__()
        self.rnn = nn.LSTM(input_size=state_dim + action_dim, hidden_size=rnn_hidden,
                           num_layers=rnn_layers, batch_first=True)
        self.ln = LayerNorm1D(rnn_hidden)
        self.fc = nn.Sequential(
            nn.Linear(rnn_hidden, fc_hidden), nn.ReLU(inplace=True),
            nn.Linear(fc_hidden, 1)
        )
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            fanin_init(m.weight); nn.init.zeros_(m.bias)

    def forward(self, s, a, h=None):
        x = torch.cat([s, a], dim=-1)
        y, h = self.rnn(x, h)
        y = self.ln(y)
        q = self.fc(y)  # [B,T,1]
        return q, h

# =========================
# Replay (sequence segments)
# =========================
class EpisodeBuf:
    def __init__(self):
        self.s, self.a, self.r, self.s2, self.d = [], [], [], [], []
    def add(self, s, a, r, s2, d):
        self.s.append(s); self.a.append(a); self.r.append(r); self.s2.append(s2); self.d.append(d)

class SeqReplay:
    def __init__(self, capacity_eps=1000, seq_len=16, device="cpu"):
        self.capacity_eps = capacity_eps
        self.seq_len = seq_len
        self.device = device
        self.buf = []

    def start_episode(self):
        self.buf.append(EpisodeBuf())
        if len(self.buf) > self.capacity_eps:
            self.buf.pop(0)

    def add(self, s, a, r, s2, d):
        if not self.buf:
            self.start_episode()
        self.buf[-1].add(s, a, r, s2, d)
        if d:
            self.start_episode()

    def __len__(self):
        return sum(len(ep.s) for ep in self.buf)

    def sample(self, batch_size):
        Ss, As, Rs, S2s, Ds, Ms = [], [], [], [], [], []
        for _ in range(batch_size):
            ep = np.random.choice(self.buf)
            while len(ep.s) < self.seq_len:
                ep = np.random.choice(self.buf)
            t0 = np.random.randint(0, len(ep.s) - self.seq_len + 1)
            t1 = t0 + self.seq_len
            s  = np.asarray(ep.s [t0:t1], np.float32)
            a  = np.asarray(ep.a [t0:t1], np.float32)
            r  = np.asarray(ep.r [t0:t1], np.float32)[..., None]
            s2 = np.asarray(ep.s2[t0:t1], np.float32)
            d  = np.asarray(ep.d [t0:t1], np.float32)[..., None]
            m  = np.ones_like(r, np.float32)
            Ss.append(s); As.append(a); Rs.append(r); S2s.append(s2); Ds.append(d); Ms.append(m)
        to_t = lambda xs: torch.tensor(np.stack(xs), device=self.device)
        return to_t(Ss), to_t(As), to_t(Rs), to_t(S2s), to_t(Ds), to_t(Ms)

# =========================
# Agent (TD3-style recurrent DDPG)
# =========================
class RDDPG:
    """
    Twin Critics + Target Policy Smoothing + Policy Delay
    Optional: burn-in and n-step targets over sequences.
    """
    def __init__(self, state_dim, action_dim, action_high,
                 actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=5e-3,
                 rnn_hidden=128, rnn_layers=1, fc_hidden=256,
                 policy_delay=2, target_noise=0.2, target_noise_clip=0.5,
                 burnin=4, n_step=1,
                 device="cpu"):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.action_high = float(action_high)

        self.policy_delay = policy_delay
        self.target_noise = target_noise
        self.target_noise_clip = target_noise_clip
        self.burnin = burnin
        self.n_step = max(1, n_step)

        # Actor + Target
        self.actor = ActorLSTM(state_dim, action_dim, action_high,
                               rnn_hidden, rnn_layers, fc_hidden).to(device)
        self.actor_tgt = copy.deepcopy(self.actor).to(device)

        # Twin Critics + Targets
        self.critic1 = CriticLSTM(state_dim, action_dim,
                                  rnn_hidden, rnn_layers, fc_hidden).to(device)
        self.critic2 = CriticLSTM(state_dim, action_dim,
                                  rnn_hidden, rnn_layers, fc_hidden).to(device)
        self.critic1_tgt = copy.deepcopy(self.critic1).to(device)
        self.critic2_tgt = copy.deepcopy(self.critic2).to(device)

        # Optims
        self.pi_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.q1_opt = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.q2_opt = optim.Adam(self.critic2.parameters(), lr=critic_lr)

        self.mse = nn.MSELoss(reduction="none")
        self.train_it = 0

    @torch.no_grad()
    def act(self, s_t, h=None, noise_std=0.1):
        """
        s_t: np.array [state_dim] or torch [state_dim]
        returns: action np.array [action_dim], new hidden state
        """
        if not torch.is_tensor(s_t):
            s_t = torch.tensor(s_t, dtype=torch.float32, device=self.device)
        s_t = s_t.unsqueeze(0).unsqueeze(1)  # [1,1,D]
        a_t, h_new = self.actor.act_last(s_t, h)
        a = a_t.squeeze(0)
        if noise_std > 0:
            a = a + noise_std * torch.randn_like(a)
        a = torch.clamp(a, -self.action_high, self.action_high)
        return a.detach().cpu().numpy(), h_new

    def _soft_update(self, net, tgt):
        with torch.no_grad():
            for p, tp in zip(net.parameters(), tgt.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

    def _apply_burnin_mask(self, M, burnin):
        if burnin <= 0:
            return M
        mask = M.clone()
        T = mask.size(1)
        mask[:, :min(burnin, T), :] = 0.0
        return mask

    def _n_step_bootstrap(self, R, D, q_next, gamma, n):
        """
        R, D, q_next: [B,T,1]
        returns y: [B,T,1]   (n-step target with truncate at done)
        """
        B, T, _ = R.shape
        y = torch.zeros_like(R)
        pow_g = torch.ones_like(R)
        acc = torch.zeros_like(R)

        for k in range(n):
            acc[:, :T-k, :] = acc[:, :T-k, :] + pow_g[:, :T-k, :] * R[:, k:, :]
            pow_g[:, :T-k, :] = pow_g[:, :T-k, :] * (gamma * (1.0 - D[:, k:, :]))

        if n - 1 < T:
            acc[:, :T-(n-1), :] = acc[:, :T-(n-1), :] + pow_g[:, :T-(n-1), :] * q_next[:, n-1:, :]
        y = acc
        return y

    def train_step(self, batch):
        """
        batch: (S, A, R, S2, D, M) with shape [B,T,...]
        """
        S, A, R, S2, D, M = batch

        # ----- Target actions with smoothing -----
        with torch.no_grad():
            a2, _ = self.actor_tgt(S2)  # [B,T,Adim]
            if self.target_noise > 0:
                eps = torch.randn_like(a2) * self.target_noise
                eps = torch.clamp(eps, -self.target_noise_clip, self.target_noise_clip)
                a2 = a2 + eps
                a2 = torch.clamp(a2, -self.action_high, self.action_high)

            q1_tgt, _ = self.critic1_tgt(S2, a2)
            q2_tgt, _ = self.critic2_tgt(S2, a2)
            q_next = torch.min(q1_tgt, q2_tgt)  # TD3 min trick

            if self.n_step > 1:
                y = self._n_step_bootstrap(R, D, q_next, self.gamma, self.n_step)
            else:
                y = R + (1.0 - D) * (self.gamma * q_next)

        # burn-in mask
        M_eff = self._apply_burnin_mask(M, self.burnin)

        # ----- Critic 1 -----
        q1, _ = self.critic1(S, A)
        loss_q1 = (self.mse(q1, y) * M_eff).sum() / M_eff.sum().clamp_min(1.0)

        self.q1_opt.zero_grad(set_to_none=True)
        loss_q1.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 10.0)
        self.q1_opt.step()

        # ----- Critic 2 -----
        q2, _ = self.critic2(S, A)
        loss_q2 = (self.mse(q2, y) * M_eff).sum() / M_eff.sum().clamp_min(1.0)

        self.q2_opt.zero_grad(set_to_none=True)
        loss_q2.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 10.0)
        self.q2_opt.step()

        # ----- Delayed policy update -----
        self.train_it += 1
        actor_loss_value = None
        if self.train_it % self.policy_delay == 0:
            a_pi, _ = self.actor(S)
            q_pi, _ = self.critic1(S, a_pi)      # فقط critic1 برای گرادیان
            actor_loss = (-(q_pi) * M_eff).sum() / M_eff.sum().clamp_min(1.0)

            self.pi_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
            self.pi_opt.step()
            actor_loss_value = float(actor_loss.item())

            # targets update
            self._soft_update(self.actor, self.actor_tgt)
            self._soft_update(self.critic1, self.critic1_tgt)
            self._soft_update(self.critic2, self.critic2_tgt)

        return {
            "loss_q1": float(loss_q1.item()),
            "loss_q2": float(loss_q2.item()),
            "actor_loss": actor_loss_value,
            "q1_mean": float(q1.mean().item()),
            "q2_mean": float(q2.mean().item()),
        }
