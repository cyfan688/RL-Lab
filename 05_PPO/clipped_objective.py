import torch


def ppo_actor_loss(
    new_log_probs,
    old_log_probs,
    advantages,
    clip_epsilon=0.2,
):
    ratio = torch.exp(new_log_probs - old_log_probs)

    surrogate_1 = ratio * advantages

    clipped_ratio = torch.clamp(
        ratio,
        1 - clip_epsilon,
        1 + clip_epsilon,
    )
    surrogate_2 = clipped_ratio * advantages

    actor_loss = -torch.min(
        surrogate_1,
        surrogate_2,
    ).mean()

    return actor_loss, ratio