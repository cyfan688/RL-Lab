import torch


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    gamma: float,
    gae_lambda: float
) -> torch.Tensor:
    """
    Compute Generalized Advantage Estimation.

    All input tensors should have shape [T].
    """
    deltas = (
        rewards
        + gamma * (1.0 - terminated) * next_values
        - values
    )

    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(
        (),
        dtype=rewards.dtype,
        device=rewards.device
    )

    for t in reversed(range(len(rewards))):
        gae = (
            deltas[t]
            + gamma
            * gae_lambda
            * (1.0 - terminated[t])
            * gae
        )

        advantages[t] = gae

    return advantages

