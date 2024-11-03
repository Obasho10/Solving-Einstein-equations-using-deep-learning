import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad


class MetricPINN(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_hidden_layers=8, output_size=10):
        super(MetricPINN, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers)])
        self.output_layer = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = F.logsigmoid(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = F.logsigmoid(hidden_layer(x))
        output = self.output_layer(x)
        g = torch.zeros((x.size(0), 4, 4), device=x.device)
        g[:, 0, 0] = output[:, 0]  # g_00
        g[:, 0, 1] = g[:, 1, 0] = output[:, 1]  # g_01
        g[:, 0, 2] = g[:, 2, 0] = output[:, 2]  # g_02
        g[:, 0, 3] = g[:, 3, 0] = output[:, 3]  # g_03
        g[:, 1, 1] = output[:, 4]  # g_11
        g[:, 1, 2] = g[:, 2, 1] = output[:, 5]  # g_12
        g[:, 1, 3] = g[:, 3, 1] = output[:, 6]  # g_13
        g[:, 2, 2] = output[:, 7]  # g_22
        g[:, 2, 3] = g[:, 3, 2] = output[:, 8]  # g_23
        g[:, 3, 3] = output[:, 9]  # g_33

        return g


