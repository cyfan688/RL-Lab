import torch

def compute_n_step_returns(
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    bootstrap_value: torch.Tensor,
    gamma: float
) -> torch.Tensor:
    returns = []

    running_return = bootstrap_value

    for t in reversed(range(len(rewards))):
        running_return = (
            rewards[t]
            + gamma
            * (1.0 - terminated[t])
            * running_return
        )

        returns.append(running_return)

    returns.reverse()

    return torch.stack(returns)