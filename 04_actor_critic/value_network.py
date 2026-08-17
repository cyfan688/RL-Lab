import torch
import torch.nn as nn

class ValueNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        value = self.net(obs)

        # 将形状从 [batch_size, 1] 变成 [batch_size]
        return value.squeeze(-1)