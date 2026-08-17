import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from value_network import ValueNetwork

#from td_learning import compute_n_step_returns
from GAE import compute_gae
from dataclasses import dataclass


@dataclass
class Rollout:
    states: list[torch.Tensor]
    rewards: list[float]
    next_states: list[torch.Tensor]
    terminated: list[bool]
    log_probs: list[torch.Tensor]

class PolicyNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 128
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class ActorCritic:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,

        #essntial hyperparameters, which determine the learning process of the agent.
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.actor = PolicyNetwork(obs_dim, act_dim)
        self.critic = ValueNetwork(obs_dim)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=actor_lr
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=critic_lr
        )

    def select_action(
        self,
        state: torch.Tensor
    ) -> tuple[int, torch.Tensor]:
        logits = self.actor(state)

        dist = Categorical(logits=logits)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob

    def collect_rollout(
        self,
        env,
        state: torch.Tensor,
        rollout_length: int
    ) -> tuple[Rollout, torch.Tensor, bool, float]:
        rollout = Rollout(
            states=[],
            rewards=[],
            next_states=[],
            terminated=[],
            log_probs=[]
        )

        episode_done = False
        collected_reward = 0.0

        for _ in range(rollout_length):
            action, log_prob = self.select_action(state)

            next_state_np, reward, terminated, truncated, _ = env.step(
                action
            )

            next_state = torch.tensor(
                next_state_np,
                dtype=torch.float32
            )

            rollout.states.append(state)
            rollout.rewards.append(float(reward))
            rollout.next_states.append(next_state)
            rollout.terminated.append(terminated)
            rollout.log_probs.append(log_prob)

            collected_reward += float(reward)
            episode_done = terminated or truncated
            state = next_state

            if episode_done:
                break

        return rollout, state, episode_done, collected_reward

    def update_batch(
    self,
    rollout: Rollout
    ) -> tuple[float, float]:
        states = torch.stack(rollout.states)
        rewards = torch.tensor(
        rollout.rewards,
        dtype=torch.float32
        )
        terminated = torch.tensor(
        rollout.terminated,
        dtype=torch.float32
        )
        log_probs = torch.stack(
        rollout.log_probs
        )
        values = self.critic(states)


        # TD(0) returns
        #with torch.no_grad():
        #    final_next_state = rollout.next_states[-1]

        #bootstrap_value = self.critic(
        #    final_next_state
        #)

        #returns = compute_n_step_returns(
        #    rewards=rewards,
        #    terminated=terminated,
        #    bootstrap_value=bootstrap_value,
        #    gamma=self.gamma
        #)
        # advantages = returns - values
        # advantages = returns - values
        # critic_loss = F.mse_loss(
        # values,
        # returns)
        next_states = torch.stack(rollout.next_states)
        with torch.no_grad():
        # next_values[t] = V(s_{t+1})
            next_values = self.critic(
                next_states
                )
            advantages = compute_gae(
            rewards=rewards,
            values=values.detach(),
            next_values=next_values,
            terminated=terminated,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda
            )
            value_targets = (
                advantages
                + values.detach()
            )
        critic_loss = F.mse_loss(
            values,
            value_targets
        )

        
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        actor_loss = -(
                log_probs * advantages.detach()
                ).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss.item(), critic_loss.item()

def train():
    env = gym.make("CartPole-v1")

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    agent = ActorCritic(
        obs_dim=obs_dim,
        act_dim=act_dim
    )

    rollout_length = 3
    num_episodes = 500

    for episode in range(num_episodes):
        state_np, _ = env.reset()

        state = torch.tensor(
            state_np,
            dtype=torch.float32
        )

        episode_done = False
        episode_return = 0.0

        while not episode_done:
            rollout, state, episode_done, collected_reward = (
                agent.collect_rollout(
                    env=env,
                    state=state,
                    rollout_length=rollout_length
                )
            )

            actor_loss, critic_loss = agent.update_batch(
                rollout
            )

            episode_return += collected_reward

        print(
            f"Episode {episode + 1}, "
            f"Return: {episode_return:.1f}, "
            f"Actor Loss: {actor_loss:.4f}, "
            f"Critic Loss: {critic_loss:.4f}"
        )

    env.close()

if __name__ == "__main__":
    train()