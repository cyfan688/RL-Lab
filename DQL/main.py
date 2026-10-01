import random
from pathlib import Path
import math

import ale_py
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from stable_baselines3.common.buffers import ReplayBuffer


# ============================================================
# 1. DQN Network
# ============================================================

class DQN(nn.Module):
    def __init__(self, nb_actions):
        super().__init__()

        self.network = nn.Sequential(
            # Input: (batch, 4, 84, 84)

            nn.Conv2d(
                in_channels=4,
                out_channels=16,
                kernel_size=8,
                stride=4
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=4,
                stride=2
            ),
            nn.ReLU(),

            nn.Flatten(),

            # 32 * 9 * 9 = 2592
            nn.Linear(2592, 256),
            nn.ReLU(),

            # One Q value for every possible action
            nn.Linear(256, nb_actions)
        )

    def forward(self, x):
        # Atari observations are uint8 in [0, 255]
        # Convert them to [0, 1]
        x = x.float() / 255.0

        return self.network(x)


# ============================================================
# 2. Build Atari Environment
# ============================================================

def make_env():
    # Gymnasium 1.x requires ALE registration
    gym.register_envs(ale_py)

    # Important:
    # frameskip=1 here because AtariPreprocessing
    # will perform frame skip = 4 later.
    env = gym.make(
        "ALE/Breakout-v5",
        frameskip=1,

        # Original DQN-style setting:
        # no sticky actions
        repeat_action_probability=0.0,
    )

    # Atari preprocessing:
    #
    # raw RGB frame
    #       ↓
    # frame skipping
    #       ↓
    # max pooling
    #       ↓
    # grayscale
    #       ↓
    # resize to 84 x 84
    #
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=False,
    )

    # (84, 84)
    #    ↓
    # stack 4 consecutive observations
    #    ↓
    # (4, 84, 84)
    env = FrameStackObservation(
        env,
        stack_size=4
    )

    return env


# ============================================================
# 3. Epsilon Schedule
# ============================================================

def get_epsilon(
    step,
    initial_exploration,
    final_exploration,
    k
):
    epsilon = (
        final_exploration
        +
        (initial_exploration - final_exploration)
        * math.exp(-k * step)
    )

    return max(
        epsilon,
        final_exploration
    )


# ============================================================
# 4. Select Action
# ============================================================

def select_action(
    obs,
    epsilon,
    q_network,
    env,
    device
):

    # ε probability:
    # random exploration
    if random.random() < epsilon:
        return env.action_space.sample()

    # Otherwise:
    # greedy action
    #
    # obs:
    # (4, 84, 84)
    #
    # unsqueeze:
    # (1, 4, 84, 84)

    obs_tensor = torch.as_tensor(
        obs,
        device=device
    ).unsqueeze(0)

    with torch.no_grad():
        q_values = q_network(obs_tensor)

    # q_values shape:
    # (1, nb_actions)

    action = q_values.argmax(
        dim=1
    ).item()

    return action


# ============================================================
# 5. DQN Training
# ============================================================

def Deep_Q_Learning(
    env,

    # NOTE:
    # 100,000 is much easier on RAM for learning/debugging.
    # Paper-scale buffers can be much larger.
    replay_memory_size=1000_000,

    total_steps=10_000_000,

    update_frequency=4,

    batch_size=128,

    discount_factor=0.99,

    replay_start_size=20_000,

    initial_exploration=1.0,

    final_exploration=0.05,

    # Target network update frequency
    target_update_frequency=10_000,

    learning_rate=1.25e-4,

    device=None,

    seed=42,
):

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = torch.device(device)

    print("Device:", device)


    # --------------------------------------------------------
    # Random seeds
    # --------------------------------------------------------

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


    # --------------------------------------------------------
    # Replay Buffer
    # --------------------------------------------------------

    rb = ReplayBuffer(
        buffer_size=replay_memory_size,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,

        # Avoid storing observation twice
        optimize_memory_usage=True,

        # We manually control terminal semantics
        handle_timeout_termination=False
    )


    # --------------------------------------------------------
    # Online Q Network
    # --------------------------------------------------------

    q_network = DQN(
        env.action_space.n
    ).to(device)


    # --------------------------------------------------------
    # Target Q Network
    # --------------------------------------------------------

    target_network = DQN(
        env.action_space.n
    ).to(device)

    # Initially:
    #
    # θ_target = θ_online

    target_network.load_state_dict(
        q_network.state_dict()
    )

    # Target network should NOT be optimized directly
    target_network.eval()


    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        q_network.parameters(),
        lr=learning_rate
    )


    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    episode_rewards = []
    smoothed_rewards = []

    Path("Imgs").mkdir(
        exist_ok=True
    )


    # --------------------------------------------------------
    # Reset environment
    # --------------------------------------------------------

    obs, info = env.reset(seed=seed)

    # Breakout requires FIRE to serve the ball.
    obs, fire_reward, terminated, truncated, info = env.step(1)

    episode_reward = float(fire_reward)

    current_lives = info.get("lives", None)


    # ========================================================
    # Main Training Loop
    # ========================================================

    progress_bar = tqdm(
        range(total_steps)
    )

    for step in progress_bar:

        # ----------------------------------------------------
        # A. epsilon
        # ----------------------------------------------------

        epsilon = get_epsilon(
            step,
            initial_exploration,
            final_exploration,
            k=5e-7
        )


        # ----------------------------------------------------
        # B. choose action
        # ----------------------------------------------------

        action = select_action(
            obs,
            epsilon,
            q_network,
            env,
            device
        )


        # ----------------------------------------------------
        # C. interact with environment
        # ----------------------------------------------------

        next_obs, reward, terminated, truncated, info = \
            env.step(action)

        episode_reward += float(reward)


        # ----------------------------------------------------
        # D. Detect loss of life
        # ----------------------------------------------------

        new_lives = info.get(
            "lives",
            current_lives
        )

        life_lost = False

        if (
            current_lives is not None
            and new_lives is not None
            and new_lives < current_lives
            and new_lives > 0
        ):
            life_lost = True


        # Actual game over
        episode_done = (
            terminated
            or truncated
        )


        # For DQN learning:
        #
        # losing a life is treated as terminal
        #
        # This prevents:
        #
        # r + gamma * Q(next_state)
        #
        # from bootstrapping across a lost life.
        learning_done = (
            terminated
            or life_lost
        )


        # ----------------------------------------------------
        # E. Reward clipping
        # ----------------------------------------------------

        clipped_reward = np.sign(
            reward
        ).astype(np.float32)


        # ----------------------------------------------------
        # F. Store transition in replay buffer
        # ----------------------------------------------------

        rb.add(
            obs,
            next_obs,

            np.array(action),

            np.array(
                clipped_reward,
                dtype=np.float32
            ),

            np.array(
                learning_done,
                dtype=np.float32
            ),

            [info]
        )


        # ----------------------------------------------------
        # G. Move state
        # ----------------------------------------------------

        obs = next_obs
        current_lives = new_lives


        # ----------------------------------------------------
        # H. FIRE again after losing a life
        # ----------------------------------------------------

        # Current ALE documentation explicitly states that
        # Breakout needs FIRE after every life loss.

        if life_lost and not episode_done:

            obs, fire_reward, fire_terminated, fire_truncated, info = \
                env.step(1)

            episode_reward += float(
                fire_reward
            )

            current_lives = info.get(
                "lives",
                current_lives
            )

            episode_done = (
                fire_terminated
                or fire_truncated
            )


        # ====================================================
        # I. Train DQN
        # ====================================================

        if (
            step >= replay_start_size
            and step % update_frequency == 0
        ):

            # -----------------------------------------------
            # Sample random minibatch
            # -----------------------------------------------

            data = rb.sample(
                batch_size
            )


            # data.observations
            #
            # shape:
            # (batch_size, 4, 84, 84)


            # ===============================================
            # TD TARGET
            # ===============================================

            with torch.no_grad():

                next_online_q_values = q_network(
                    data.next_observations
                )

                next_actions = next_online_q_values.argmax(
                    dim=1,
                    keepdim=True
                )

                next_target_q_values = target_network(
                    data.next_observations
                )

                next_q = (
                    next_target_q_values
                    .gather(1, next_actions)
                    .squeeze(1)
                )

                rewards = (
                    data.rewards
                    .flatten()
                )

                dones = (
                    data.dones
                    .flatten()
                )

                # y =
                #
                # r
                # +
                # gamma
                # *
                # (1-done)
                # *
                # max Q_target(s',a')

                td_target = (
                    rewards
                    +
                    discount_factor
                    * (1 - dones)
                    * next_q
                )


            # ===============================================
            # CURRENT Q
            # ===============================================

            # Online network outputs:
            #
            # [
            #   Q(s,0),
            #   Q(s,1),
            #   ...
            # ]

            all_current_q = q_network(
                data.observations
            )


            # But transition only contains one actual action.
            #
            # gather selects:
            #
            # Q(s,a)

            current_q = (
                all_current_q
                .gather(
                    dim=1,
                    index=data.actions.long()
                )
                .squeeze(1)
            )


            # ===============================================
            # LOSS
            # ===============================================

            loss = F.smooth_l1_loss(
                current_q,
                td_target
            )

            losses.append(loss.item())


            # ===============================================
            # SGD / Adam update
            # ===============================================

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()


        # ====================================================
        # J. Update Target Network
        # ====================================================

        if (
            step > replay_start_size
            and step % target_update_frequency == 0
        ):

            target_network.load_state_dict(
                q_network.state_dict()
            )


        # ====================================================
        # K. Episode finished
        # ====================================================

        if episode_done:

            episode_rewards.append(
                episode_reward
            )

            obs, info = env.reset()

            # Breakout needs FIRE
            obs, fire_reward, terminated, truncated, info = \
                env.step(1)

            episode_reward = float(
                fire_reward
            )

            current_lives = info.get(
                "lives",
                None
            )


        # ====================================================
        # L. Logging
        # ====================================================

        if (
            step > 0
            and step % 2000 == 0
            and len(episode_rewards) > 0
        ):

            recent_reward = np.mean(
                episode_rewards[-20:]
            )

            smoothed_rewards.append(
                recent_reward
            )

            plt.figure()

            plt.plot(
                smoothed_rewards
            )

            plt.title(
                "DQN on Breakout"
            )

            plt.xlabel(
                "2k Environment Steps"
            )

            plt.ylabel(
                "Average Episode Reward"
            )

            plt.savefig(
                "Imgs/average_reward_on_breakout.png"
            )

            plt.close()


        # Progress display
        if len(episode_rewards) > 0:

            progress_bar.set_postfix(
                epsilon=f"{epsilon:.3f}",
                reward=f"{np.mean(episode_rewards[-10:]):.1f}"
            )


    return (
        q_network,
        episode_rewards,
        losses
    )

def moving_average(data, window_size = 20):
    if len(data) < window_size:
        return np.array(data)
    return np.convolve(
        data,
        np.ones(window_size) / window_size,
        mode="valid"
    )

# ============================================================
# 6. Run
# ============================================================

if __name__ == "__main__":

    losses = []

    env = make_env()

    print(
        "Observation space:",
        env.observation_space
    )

    print(
        "Action space:",
        env.action_space
    )

    q_network, rewards = Deep_Q_Learning(
        env
    )

    env.close()

    smooth_rewards = moving_average(
        rewards,
        window=20
    )

    plt.figure()

    plt.plot(
        rewards,
        alpha=0.3,
        label="Raw Reward"
    )

    if len(rewards) >= 20:
        plt.plot(
            range(19, len(rewards)),
            smooth_rewards,
            label="20-Episode Moving Average"
        )

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DQN on Breakout")
    plt.legend()

    plt.show()


    # ==========================
    # Loss Curve
    # ==========================

    plt.figure()

    plt.plot(losses)

    plt.xlabel("Gradient Update")
    plt.ylabel("Huber Loss")
    plt.title("DQN Training Loss")

    plt.show()
