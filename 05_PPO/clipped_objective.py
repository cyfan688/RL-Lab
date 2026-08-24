import torch

def ppo_clipped_loss(
    new_log_probs,
    old_log_probs,
    advantages,
    clip_epsilon=0.2,
):
    # π_new(a|s) / π_old(a|s)
    ratio = torch.exp(new_log_probs - old_log_probs)

    original_objective = ratio * advantages

    clipped_ratio = torch.clamp(
        ratio,
        1 - clip_epsilon,
        1 + clip_epsilon,
    )
    clipped_objective = clipped_ratio * advantages

    # PPO chooses the more conservative objective
    objective = torch.minimum(
        original_objective,
        clipped_objective,
    )

    # PyTorch minimizes loss, while PPO maximizes objective
    actor_loss = -objective.mean()

    return actor_loss, ratio, original_objective, clipped_objective, objective

# if __name__ == "__main__":
#     old_probs = torch.tensor([0.4, 0.4, 0.4, 0.4])
#     new_probs = torch.tensor([0.6, 0.2, 0.2, 0.6])

#     old_log_probs = torch.log(old_probs)
#     new_log_probs = torch.log(new_probs)

#     # Four combinations:
#     # 1. A > 0, ratio > 1.2
#     # 2. A > 0, ratio < 0.8
#     # 3. A < 0, ratio < 0.8
#     # 4. A < 0, ratio > 1.2
#     advantages = torch.tensor([1.0, 1.0, -1.0, -1.0])

#     (
#         actor_loss,
#         ratio,
#         original,
#         clipped,
#         objective,
#     ) = ppo_clipped_loss(
#         new_log_probs,
#         old_log_probs,
#         advantages,
#     )

#     print("ratio:             ", ratio)
#     print("advantages:        ", advantages)
#     print("original objective:", original)
#     print("clipped objective: ", clipped)
#     print("chosen objective:  ", objective)
#     print("actor loss:        ", actor_loss)

#     expected = torch.tensor([1.2, 0.5, -0.8, -1.5])
#     assert torch.allclose(objective, expected, atol=1e-6)

#     print("Clipped objective test passed!")