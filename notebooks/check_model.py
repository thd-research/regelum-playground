import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class Actor(nn.Module):
    def __init__(
        self,
        dim_action: int,
        dim_observation: int,
        action_bounds: list[list[float]],
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.fc1 = nn.Linear(dim_observation, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, dim_action)
        self.fc_logstd = nn.Linear(256, dim_action)
        self.action_bounds = np.array(action_bounds)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (action_bounds[:, 1] - action_bounds[:, 0]) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (action_bounds[:, 1] + action_bounds[:, 0]) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean