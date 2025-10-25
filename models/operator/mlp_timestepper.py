import torch
import torch.nn as nn

class TimeStepperMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, 
                 activation, output_dim):
        super(TimeStepperMLP, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation)
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, state, param):
        x = torch.cat([state, param], dim=-1)
        return self.net(x)