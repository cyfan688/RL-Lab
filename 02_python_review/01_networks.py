import torch
import torch.nn as nn

layer = nn.Linear(4, 2)

print(layer.weight)
print(layer.bias)