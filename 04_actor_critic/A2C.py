
from collections import deque
from dataclasses import dataclass, field
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.vector import AutoresetMode, SyncVectorEnv
from torch.distributions import Categorical


@dataclass
class Config:
    env_id: str = "CartPole-v1"
    seed: int = 42

    # A2C: N 个环境，每个环境连续收集 T 步
    num_envs: int = 8
    rollout_length: int = 5
    num_updates: int = 2_000

    gamma: float = 0.99
    gae_lambda: float = 0.95

    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    normalize_advantage: bool = True
    print_every: int = 20
    hidden_dim: int = 128


@dataclass
class Rollout:
    # 每个列表长度都是 T；列表中的每个 tensor 都含 N 个环境的数据
    states: list[torch.Tensor] = field(default_factory=list)
    next_states: list[torch.Tensor] = field(default_factory=list)
    rewards: list[torch.Tensor] = field(default_factory=list)
    terminated: list[torch.Tensor] = field(default_factory=list)
    dones: list[torch.Tensor] = field(default_factory=list)
    log_probs: list[torch.Tensor] = field(default_factory=list)
    entropies: list[torch.Tensor] = field(default_factory=list)


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        # 单环境: [obs_dim] -> [act_dim]
        # 多环境: [N, obs_dim] -> [N, act_dim]
        return self.net(states)


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        # [T, N, obs_dim] -> [T, N]
        return self.net(states).squeeze(-1)


class A2CAgent:
    def __init__(self, obs_dim: int, act_dim: int, config: Config):
        self.config = config

        self.actor = PolicyNetwork(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dim=config.hidden_dim,
        )
        self.critic = ValueNetwork(
            obs_dim=obs_dim,
            hidden_dim=config.hidden_dim,
        )

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=config.actor_lr,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=config.critic_lr,
        )

    def select_actions(
        self,
        states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        states:     [N, obs_dim]
        actions:    [N]
        log_probs:  [N]
        entropies:  [N]
        """
        logits = self.actor(states)
        distribution = Categorical(logits=logits)

        actions = distribution.sample()
        log_probs = distribution.log_prob(actions)
        entropies = distribution.entropy()

        return actions, log_probs, entropies

    def collect_rollout(
        self,
        envs: SyncVectorEnv,
        states: torch.Tensor,
        running_episode_returns: np.ndarray,
    ) -> tuple[Rollout, torch.Tensor, list[float]]:
        """让 N 个环境同步向前走 T 步，并保存 N×T 个 transition。"""
        rollout = Rollout()
        completed_returns: list[float] = []

        for _ in range(self.config.rollout_length):
            actions, log_probs, entropies = self.select_actions(states)

            (
                next_states_np,
                rewards_np,
                terminated_np,
                truncated_np,
                _,
            ) = envs.step(actions.detach().cpu().numpy())

            # DISABLED autoreset 模式下，这里仍然是真正的 episode 末状态，
            # 因而 truncation 时可以从它 bootstrap。
            transition_next_states = torch.tensor(
                next_states_np,
                dtype=torch.float32,
            )
            rewards = torch.tensor(rewards_np, dtype=torch.float32)
            terminated = torch.tensor(terminated_np, dtype=torch.float32)

            done_np = np.logical_or(terminated_np, truncated_np)
            dones = torch.tensor(done_np, dtype=torch.float32)

            rollout.states.append(states)
            rollout.next_states.append(transition_next_states)
            rollout.rewards.append(rewards)
            rollout.terminated.append(terminated)
            rollout.dones.append(dones)
            rollout.log_probs.append(log_probs)
            rollout.entropies.append(entropies)

            # 只用于打印 episode return，不参与训练
            running_episode_returns += rewards_np
            finished_indices = np.flatnonzero(done_np)
            for env_index in finished_indices:
                completed_returns.append(
                    float(running_episode_returns[env_index])
                )
                running_episode_returns[env_index] = 0.0

            # 某些子环境结束后，只重置这些环境。
            if np.any(done_np):
                reset_states_np, _ = envs.reset(
                    options={"reset_mask": done_np}
                )
                states = torch.tensor(
                    reset_states_np,
                    dtype=torch.float32,
                )
            else:
                states = transition_next_states

        return rollout, states, completed_returns

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        terminated: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """
        所有输入形状均为 [T, N]。

        terminated 控制 bootstrap：真正终止后 V=0；时间上限 truncation 仍可 bootstrap。
        dones 控制递推边界：无论 terminated 还是 truncated，都不能跨 episode 累积 GAE。
        """
        deltas = (
            rewards
            + self.config.gamma * (1.0 - terminated) * next_values
            - values
        )

        advantages = torch.zeros_like(rewards)
        gae = torch.zeros_like(rewards[0])  # [N]

        for t in reversed(range(rewards.shape[0])):
            gae = (
                deltas[t]
                + self.config.gamma
                * self.config.gae_lambda
                * (1.0 - dones[t])
                * gae
            )
            advantages[t] = gae

        return advantages

    def update(self, rollout: Rollout) -> tuple[float, float, float]:
        """对 N×T 个样本求平均梯度，并同步更新 Actor 与 Critic。"""
        # states: [T, N, obs_dim]
        states = torch.stack(rollout.states)
        next_states = torch.stack(rollout.next_states)

        # 其余张量: [T, N]
        rewards = torch.stack(rollout.rewards)
        terminated = torch.stack(rollout.terminated)
        dones = torch.stack(rollout.dones)
        log_probs = torch.stack(rollout.log_probs)
        entropies = torch.stack(rollout.entropies)

        values = self.critic(states)

        with torch.no_grad():
            next_values = self.critic(next_states)

            advantages = self.compute_gae(
                rewards=rewards,
                values=values.detach(),
                next_values=next_values,
                terminated=terminated,
                dones=dones,
            )

            value_targets = advantages + values.detach()

        critic_loss = F.mse_loss(values, value_targets)

        actor_advantages = advantages.detach()
        if self.config.normalize_advantage and actor_advantages.numel() > 1:
            actor_advantages = (
                actor_advantages - actor_advantages.mean()
            ) / (
                actor_advantages.std(unbiased=False) + 1e-8
            )

        # mean 同时平均 T 个时间步和 N 个环境
        policy_loss = -(log_probs * actor_advantages).mean()
        entropy = entropies.mean()
        actor_loss = policy_loss - self.config.entropy_coef * entropy

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            self.config.max_grad_norm,
        )
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            self.config.max_grad_norm,
        )
        self.actor_optimizer.step()

        return actor_loss.item(), critic_loss.item(), entropy.item()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(env_id: str):
    def thunk():
        return gym.make(env_id)

    return thunk


def train() -> None:
    config = Config()
    set_seed(config.seed)

    envs = SyncVectorEnv(
        [make_env(config.env_id) for _ in range(config.num_envs)],
        autoreset_mode=AutoresetMode.DISABLED,
    )

    obs_dim = int(envs.single_observation_space.shape[0])
    act_dim = int(envs.single_action_space.n)

    agent = A2CAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        config=config,
    )

    states_np, _ = envs.reset(seed=config.seed)
    states = torch.tensor(states_np, dtype=torch.float32)

    running_episode_returns = np.zeros(
        config.num_envs,
        dtype=np.float32,
    )
    recent_returns: deque[float] = deque(maxlen=100)

    try:
        for update_index in range(1, config.num_updates + 1):
            rollout, states, completed_returns = agent.collect_rollout(
                envs=envs,
                states=states,
                running_episode_returns=running_episode_returns,
            )
            recent_returns.extend(completed_returns)

            actor_loss, critic_loss, entropy = agent.update(rollout)

            if update_index % config.print_every == 0:
                total_steps = (
                    update_index
                    * config.num_envs
                    * config.rollout_length
                )
                mean_return = (
                    float(np.mean(recent_returns))
                    if recent_returns
                    else 0.0
                )

                print(
                    f"Update: {update_index:4d} | "
                    f"Steps: {total_steps:7d} | "
                    f"Mean Return(100): {mean_return:7.2f} | "
                    f"Actor Loss: {actor_loss:8.4f} | "
                    f"Critic Loss: {critic_loss:8.4f} | "
                    f"Entropy: {entropy:7.4f}"
                )
    finally:
        envs.close()


if __name__ == "__main__":
    train()