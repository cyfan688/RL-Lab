import torch
import torch.nn as nn
from torch.distributions import Categorical

import gymnasium as gym

from collections import deque


class ActorCritic(nn.Module):
    def __init__(self, state_dim=4, action_dim=2):
        super().__init__()

        self.shared_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # Actor outputs action logits
        self.actor_head = nn.Linear(128, action_dim)

        # Critic outputs V(s)
        self.critic_head = nn.Linear(128, 1)

    def forward(self, states):
        features = self.shared_network(states)

        action_logits = self.actor_head(features)

        # Shape: (batch_size, 1) -> (batch_size,)
        values = self.critic_head(features).squeeze(-1)

        return action_logits, values

    def get_action_and_value(self, states, actions=None):
        action_logits, values = self.forward(states)

        distribution = Categorical(logits=action_logits)

        # During rollout: sample a new action
        if actions is None:
            actions = distribution.sample()

        # During PPO update:
        # calculate the new log probability of the stored action
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()

        return actions, log_probs, entropy, values




def collect_rollout(
    env,
    model,
    state,
    episode_return,
    rollout_steps=128,
):
    states = []
    actions = []
    rewards = []
    next_states = []

    terminateds = []
    episode_ends = []

    old_log_probs = []
    values = []

    completed_episode_returns = []

    for _ in range(rollout_steps):
        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
        ).unsqueeze(0)

        # Collect data without constructing a computation graph
        with torch.no_grad():
            (
                action,
                old_log_prob,
                _,
                value,
            ) = model.get_action_and_value(state_tensor)

        next_state, reward, terminated, truncated, _ = (
            env.step(action.item())
        )

        episode_end = terminated or truncated
        episode_return += reward

        states.append(state_tensor.squeeze(0))
        actions.append(action.squeeze(0))
        rewards.append(float(reward))

        next_states.append(
            torch.as_tensor(
                next_state,
                dtype=torch.float32,
            )
        )

        terminateds.append(float(terminated))
        episode_ends.append(float(episode_end))

        old_log_probs.append(old_log_prob.squeeze(0))
        values.append(value.squeeze(0))

        if episode_end:
            completed_episode_returns.append(episode_return)

            state, _ = env.reset()
            episode_return = 0.0
        else:
            state = next_state

    rollout = {
        "states": torch.stack(states),
        "actions": torch.stack(actions),
        "rewards": torch.tensor(
            rewards,
            dtype=torch.float32,
        ),
        "next_states": torch.stack(next_states),
        "terminateds": torch.tensor(
            terminateds,
            dtype=torch.float32,
        ),
        "episode_ends": torch.tensor(
            episode_ends,
            dtype=torch.float32,
        ),
        "old_log_probs": torch.stack(old_log_probs),
        "values": torch.stack(values),
    }

    return (
        rollout,
        state,
        episode_return,
        completed_episode_returns,
    )



def compute_gae(
    model,
    rollout,
    gamma=0.95,
    gae_lambda=0.95,
):
    rewards = rollout["rewards"]
    values = rollout["values"]
    next_states = rollout["next_states"]

    terminateds = rollout["terminateds"]
    episode_ends = rollout["episode_ends"]

    # Compute V(s_{t+1})
    with torch.no_grad():
        _, next_values = model(next_states)

    # A real terminal state cannot bootstrap.
    bootstrap_masks = 1.0 - terminateds

    deltas = (
        rewards
        + gamma * next_values * bootstrap_masks
        - values
    )

    advantages = torch.zeros_like(rewards)

    gae = torch.tensor(0.0)

    # Calculate backward from the final transition
    for t in reversed(range(len(rewards))):
        continuation_mask = 1.0 - episode_ends[t]

        gae = (
            deltas[t]
            + gamma
            * gae_lambda
            * continuation_mask
            * gae
        )

        advantages[t] = gae

    # Critic training target
    returns = advantages + values

    return advantages, returns, deltas



def ppo_update(
    model,
    optimizer,
    rollout,
    update_epochs=4,
    minibatch_size=64,
    clip_epsilon=0.2,
    value_coefficient=0.5,
    entropy_coefficient=0.01,
    max_grad_norm=0.5,
):
    states = rollout["states"]
    actions = rollout["actions"]
    old_log_probs = rollout["old_log_probs"]
    returns = rollout["returns"]
    advantages = rollout["advantages"]

    # Normalize only the actor's advantages
    advantages = (
        advantages - advantages.mean()
    ) / (advantages.std() + 1e-8)

    rollout_size = states.shape[0]

    metric_sums = {
        "total_loss": 0.0,
        "actor_loss": 0.0,
        "critic_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
    }

    number_of_minibatches = 0

    # Reuse the same rollout for several epochs
    for _ in range(update_epochs):
        shuffled_indices = torch.randperm(rollout_size)

        for start in range(0, rollout_size, minibatch_size):
            end = start + minibatch_size

            minibatch_indices = shuffled_indices[start:end]

            minibatch_states = states[minibatch_indices]
            minibatch_actions = actions[minibatch_indices]

            minibatch_old_log_probs = (
                old_log_probs[minibatch_indices]
            )

            minibatch_advantages = advantages[minibatch_indices]
            minibatch_returns = returns[minibatch_indices]

            # Important:
            # use the stored actions instead of sampling new actions
            (
                _,
                new_log_probs,
                entropy,
                new_values,
            ) = model.get_action_and_value(
                minibatch_states,
                minibatch_actions,
            )

            log_ratio = (
                new_log_probs
                - minibatch_old_log_probs
            )
            ratio = torch.exp(log_ratio)

            original_objective = (
                ratio * minibatch_advantages
            )

            clipped_objective = (
                torch.clamp(
                    ratio,
                    1 - clip_epsilon,
                    1 + clip_epsilon,
                )
                * minibatch_advantages
            )

            actor_loss = -torch.minimum(
                original_objective,
                clipped_objective,
            ).mean()

            critic_loss = torch.mean(
                (
                    new_values
                    - minibatch_returns
                ) ** 2
            )

            entropy_bonus = entropy.mean()

            total_loss = (
                actor_loss
                + value_coefficient * critic_loss
                - entropy_coefficient * entropy_bonus
            )

            optimizer.zero_grad()
            total_loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_grad_norm,
            )

            optimizer.step()

            # Diagnostics only
            with torch.no_grad():
                approximate_kl = (
                    ratio - 1.0 - log_ratio
                ).mean()

                clip_fraction = (
                    torch.abs(ratio - 1.0)
                    > clip_epsilon
                ).float().mean()

            metric_sums["total_loss"] += total_loss.item()
            metric_sums["actor_loss"] += actor_loss.item()
            metric_sums["critic_loss"] += critic_loss.item()
            metric_sums["entropy"] += entropy_bonus.item()
            metric_sums["approx_kl"] += approximate_kl.item()
            metric_sums["clip_fraction"] += clip_fraction.item()

            number_of_minibatches += 1

    metrics = {
        name: value / number_of_minibatches
        for name, value in metric_sums.items()
    }

    return metrics




def train_ppo(
    total_updates=300,
    rollout_steps=256,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    seed=42,
):
    torch.manual_seed(seed)

    env = gym.make("CartPole-v1")

    model = ActorCritic()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    state, _ = env.reset(seed=seed)
    episode_return = 0.0

    recent_returns = deque(maxlen=50)
    total_environment_steps = 0

    for update in range(1, total_updates + 1):
        # 1. Collect fresh on-policy data
        (
            rollout,
            state,
            episode_return,
            completed_returns,
        ) = collect_rollout(
            env=env,
            model=model,
            state=state,
            episode_return=episode_return,
            rollout_steps=rollout_steps,
        )

        total_environment_steps += rollout_steps
        recent_returns.extend(completed_returns)

        # 2. Compute GAE and critic targets
        advantages, returns, _ = compute_gae(
            model=model,
            rollout=rollout,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        rollout["advantages"] = advantages
        rollout["returns"] = returns

        # 3. Reuse this rollout for several PPO epochs
        metrics = ppo_update(
            model=model,
            optimizer=optimizer,
            rollout=rollout,
            update_epochs=4,
            minibatch_size=64,
            clip_epsilon=0.2,
            value_coefficient=0.5,
            entropy_coefficient=0.01,
            max_grad_norm=0.5,
        )

        # 4. Print training progress
        if update % 10 == 0:
            if len(recent_returns) > 0:
                average_return = (
                    sum(recent_returns)
                    / len(recent_returns)
                )
            else:
                average_return = 0.0

            print(
                f"Update: {update:4d} | "
                f"Steps: {total_environment_steps:7d} | "
                f"Average return: {average_return:7.2f} | "
                f"Actor loss: {metrics['actor_loss']:8.4f} | "
                f"Critic loss: {metrics['critic_loss']:8.4f} | "
                f"Entropy: {metrics['entropy']:7.4f} | "
                f"KL: {metrics['approx_kl']:8.5f} | "
                f"Clip fraction: "
                f"{metrics['clip_fraction']:6.3f}"
            )

        # Our early-stopping condition
        if (
            len(recent_returns) == recent_returns.maxlen
            and sum(recent_returns) / len(recent_returns)
            >= 475
        ):
            print(
                f"Environment solved at update {update}!"
            )
            break

    env.close()

    return model


def evaluate_policy(
    model,
    number_of_episodes=10,
    seed=1000,
):
    env = gym.make("CartPole-v1")

    evaluation_returns = []

    model.eval()

    for episode in range(number_of_episodes):
        state, _ = env.reset(seed=seed + episode)

        episode_return = 0.0
        episode_end = False

        while not episode_end:
            state_tensor = torch.as_tensor(
                state,
                dtype=torch.float32,
            ).unsqueeze(0)

            with torch.no_grad():
                action_logits, _ = model(state_tensor)

                # Deterministic evaluation
                action = torch.argmax(
                    action_logits,
                    dim=-1,
                ).item()

            (
                state,
                reward,
                terminated,
                truncated,
                _,
            ) = env.step(action)

            episode_return += reward
            episode_end = terminated or truncated

        evaluation_returns.append(episode_return)

    env.close()

    model.train()

    return evaluation_returns




if __name__ == "__main__":
    trained_model = train_ppo()

    evaluation_returns = evaluate_policy(
        model=trained_model,
        number_of_episodes=10,
    )

    average_evaluation_return = (
        sum(evaluation_returns)
        / len(evaluation_returns)
    )

    print("\nEvaluation returns:")
    print(evaluation_returns)

    print(
        "Average evaluation return:",
        average_evaluation_return,
    )