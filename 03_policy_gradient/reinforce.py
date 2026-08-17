import gymnasium as gym
import torch
import torch.optim as optim

from torch.distributions import Categorical
from policy_network import PolicyNetwork

def train():

    #env = gym.make("CartPole-v1")
    env = gym.make("CartPole-v1", max_episode_steps=1000)
    policy = PolicyNetwork()

    optimizer = optim.Adam(
        policy.parameters(),
        lr=1e-3
    )

    for episode in range(500):

        log_probs, rewards = rollout(env, policy)

        returns = compute_returns(rewards)

        loss = update(
            policy,
            optimizer,
            log_probs,
            returns
        )

        print(
            f"Episode {episode:3d} | "
            f"Reward = {sum(rewards):4.0f} | "
            f"Loss = {loss:.3f}"
        )

    env.close()

def rollout(env, policy):
    state, _ = env.reset()
    
    log_probs = []
    rewards = []
    done = False

    while not done:
        state_t = torch.tensor(state, dtype=torch.float32)
        logits = policy(state_t)
        dist = Categorical(logits=logits)
        
        action = dist.sample()
        log_prob = dist.log_prob(action)

        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        log_probs.append(log_prob)
        rewards.append(reward)

        state = next_state

    return log_probs, rewards


def compute_returns(rewards, gamma=0.99):
    returns = []

    G = 0

    #why?
    #Beacause the things happened before the current time step are independent of the current action.
    #So we only need to consider the rewards that come after the current time step when calculating the return for that time step. 
    # 
    # 
    for reward in reversed(rewards):
        G = reward + gamma * G

        #returns.insert(0,G)
        returns.append(G)
    returns.reverse()

    returns = torch.tensor(returns, dtype=torch.float32)

    #Here, advantages = returns
    #In actor-critic methods, advantages = returns - baseline (value function)
    

    #Centering and normalizing(industry) the returns to have mean 0 and std 1, which can help with training stability
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    return returns


def update(policy, optimizer,log_probs, returns):


    log_probs_tensor = torch.stack(log_probs)
    loss = -(log_probs_tensor * returns).mean()




    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()



if __name__ == "__main__":
    train()