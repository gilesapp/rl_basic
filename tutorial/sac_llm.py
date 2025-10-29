import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
from collections import deque
import random
import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 工具 ----------
def moving_average(x, window=50):
    return np.convolve(x, np.ones(window)/window, mode='valid')

# ---------- 缓存 ----------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)
    def store(self, s, a, r, s_, d):
        self.buf.append((s, a, r, s_, d))
    def sample(self, batch):
        batch = random.sample(self.buf, batch)
        return map(torch.tensor, zip(*batch))

# ---------- 网络 ----------
class MLP(nn.Module):
    def __init__(self, d_in, d_out, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, d_out)
        )
    def forward(self, x):
        return self.net(x)

# Actor：高斯策略 + tanh 压缩 + 重参数
class Actor(nn.Module):
    def __init__(self, s_dim, a_dim, a_bound=1.0):
        super().__init__()
        self.backbone = MLP(s_dim, 256)
        self.mu     = nn.Linear(256, a_dim)
        self.log_std = nn.Linear(256, a_dim)
        self.a_bound = a_bound
    def dist(self, s):
        h = self.backbone(s)
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), -20, 2)
        return Normal(mu, log_std.exp())
    def forward(self, s, deterministic=False):
        dist = self.dist(s)
        if deterministic:
            a = dist.mean
        else:
            a = dist.rsample()  # 重参数采样
        a = torch.tanh(a) * self.a_bound
        log_pi = dist.log_prob(a/self.a_bound).sum(-1, keepdim=True)
        log_pi -= (2*(np.log(2) - a - F.softplus(-2*a))).sum(-1, keepdim=True)
        return a, log_pi

# Critic：双 Q
class Critic(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.q1 = MLP(s_dim+a_dim, 1)
        self.q2 = MLP(s_dim+a_dim, 1)
    def both(self, s, a):
        sa = torch.cat([s, a], -1)
        return self.q1(sa), self.q2(sa)
    def forward(self, s, a):
        return torch.min(*self.both(s, a))

# ---------- SAC Agent ----------
class SAC:
    def __init__(self, s_dim, a_dim, a_bound=2.0,
                 lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2,
                 auto_alpha=True, target_ent=-2.0,
                 batch=256, buffer=1_000_000):
        self.actor = Actor(s_dim, a_dim, a_bound).to(device)
        self.critic = Critic(s_dim, a_dim).to(device)
        self.critic_tgt = Critic(s_dim, a_dim).to(device)
        self.critic_tgt.load_state_dict(self.critic.state_dict())

        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=lr) if auto_alpha else None
        self.target_ent = target_ent if target_ent else -a_dim

        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

        self.buf = ReplayBuffer(buffer)
        self.gamma, self.tau, self.batch = gamma, tau, batch
        self.a_dim, self.a_bound = a_dim, a_bound

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, s, deterministic=False):
        with torch.no_grad():
            a, _ = self.actor(torch.tensor(s, device=device, dtype=torch.float32), deterministic)
        return a.cpu().numpy()

    def update(self):
        if len(self.buf.buf) < self.batch:
            return
        s, a, r, s_, d = self.buf.sample(self.batch)
        s, a, r, s_, d = [x.to(device).float() for x in (s, a, r, s_, d)]
        r, d = r.unsqueeze(1), d.unsqueeze(1)

        # ----- Critic loss -----
        with torch.no_grad():
            a_next, log_pi_next = self.actor(s_)
            q_next = self.critic_tgt(s_, a_next) - self.alpha * log_pi_next
            y = r + self.gamma * (1-d) * q_next
        q1, q2 = self.critic.both(s, a)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ----- Actor loss -----
        a_pred, log_pi = self.actor(s)
        q_pred = self.critic(s, a_pred)
        actor_loss = (self.alpha * log_pi - q_pred).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ----- Alpha loss -----
        if self.alpha_opt:
            alpha_loss = -(self.alpha * (log_pi + self.target_ent).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

        # ----- Soft update -----
        for param, tgt in zip(self.critic.parameters(), self.critic_tgt.parameters()):
            tgt.data.copy_(self.tau * param + (1-self.tau) * tgt)

# ---------- 训练 ----------
env = gym.make('Pendulum-v1')
s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high[0]

agent = SAC(s_dim, a_dim, a_bound=a_bound, auto_alpha=True)
ep_returns = []

for ep in tqdm.trange(200):
    s, _ = env.reset()
    ep_ret = 0
    while True:
        a = agent.act(s)
        s_, r, d, trunc, _ = env.step(a)
        agent.buf.store(s, a, r, s_, d or trunc)
        agent.update()
        s = s_
        ep_ret += r
        if d or trunc:
            break
    ep_returns.append(ep_ret)
    if ep % 50 == 0:
        print(ep_ret)

# 画图
import matplotlib.pyplot as plt
plt.plot(moving_average(ep_returns))
plt.xlabel('Episode'); plt.ylabel('Return'); plt.show()