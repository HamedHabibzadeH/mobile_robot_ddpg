# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorLSTM(nn.Module):
    """
    شبکه Actor با LSTM برای LSTM-DDPG
    ورودی: state
    خروجی: action در بازه [-1, 1]
    """
    def __init__(self, obs_dim, act_dim, hidden_size=256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, act_dim)
        self.tanh = nn.Tanh()

    def forward(self, obs, hidden):
        # obs شکل: (batch, obs_dim) یا (batch, seq, obs_dim)
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        x = F.relu(self.fc1(obs))
        x, hidden = self.lstm(x, hidden)
        x = F.relu(self.fc2(x))
        a = self.tanh(self.out(x))
        return a.squeeze(1), hidden

    def init_hidden(self, batch_size=1):
        h = torch.zeros(1, batch_size, 256)
        c = torch.zeros(1, batch_size, 256)
        return h, c


class CriticLSTM(nn.Module):
    """
    شبکه Critic با LSTM برای LSTM-DDPG
    ورودی: state + action
    خروجی: Q-value
    """
    def __init__(self, obs_dim, act_dim, hidden_size=256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + act_dim, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.q_out = nn.Linear(hidden_size, 1)

    def forward(self, obs, act, hidden):
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
            act = act.unsqueeze(1)
        x = torch.cat([obs, act], dim=-1)
        x = F.relu(self.fc1(x))
        x, hidden = self.lstm(x, hidden)
        x = F.relu(self.fc2(x))
        q = self.q_out(x)
        return q.squeeze(1), hidden

    def init_hidden(self, batch_size=1):
        h = torch.zeros(1, batch_size, 256)
        c = torch.zeros(1, batch_size, 256)
        return h, c
