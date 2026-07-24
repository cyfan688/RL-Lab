import torch
import torch.nn as nn
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(4,128), nn.ReLU(),
            nn.Linear(128,128), nn.ReLU(),
            nn.Linear(128,2)
        )
   
    def forward(self, state):
        return self.network(state)

if __name__ == "__main__":
    policy = PolicyNetwork()

    state = torch.tensor( [0.02, -0.31, 0.04, 0.17],
        dtype=torch.float32)

    logits = policy(state)

    print("Logits:", logits)

    dist = Categorical(logits=logits)

    print("Probabilities:", dist.probs)

    action = dist.sample()

    print("Sampled Action:", action)