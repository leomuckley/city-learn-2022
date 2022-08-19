import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class SingleCritic(nn.Module):

    def __init__(self,
                input_size, 
                action_size = 1,
                hidden_layer_size = 32):
        super(SingleCritic, self).__init__()

        self.input_size = input_size
        self.action_size = action_size

        self._in_mlp = nn.Sequential(
            nn.Linear(input_size + action_size, hidden_layer_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_layer_size, 1),
        )

    def forward (self, state, action):
        x = torch.cat((torch.flatten(state,start_dim=1),torch.flatten(action,start_dim=1)),dim=1)
        return self._in_mlp(x)