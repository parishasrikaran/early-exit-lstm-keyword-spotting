import torch
import torch.nn as nn

class StopHead(nn.Module):
    def __init__(self, input_size=4, hidden_size=32):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)