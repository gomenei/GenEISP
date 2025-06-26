import torch
import torch.nn as nn
import torch.nn.functional as F

# use NeRF net as our model
# 
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, output_ch=4, skips=[4], CONF={}, Mx=64):
        super(NeRF, self).__init__()

        self.D = D  # Number of layers
        self.W = W  # Number of units per layer
        self.input_ch = input_ch
        self.skips = skips
        self.poslearnable = False
        self.config = CONF
        self.Mx = Mx
        if "poslearnable" in CONF['experiment'] and CONF['experiment']['poslearnable']:
            self.poslearnable = True
            self.posdim = 2
            self.pos_encoding = nn.Parameter(torch.randn(Mx, Mx, 2))    

            self.new_input_ch = self.input_ch - 2 + self.posdim
            self.input_layer = nn.Linear(self.new_input_ch, W)
        else:
        # Define the layers of the MLP network
            self.input_layer = nn.Linear(self.input_ch, W)
        self.hidden_layers = nn.ModuleList([nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        self.bn1 = nn.BatchNorm1d(W)
        self.output_layer = nn.Linear(W, output_ch)  # Output layer for RGB and density
        
    def forward(self, x):
        # Forward pass through the network
        if self.poslearnable:
            input_pts, input_xy = torch.split(x, [self.input_ch - 2, 2], dim=-1)
            input_xy = input_xy + self.pos_encoding[input_xy[:, 0].long(), input_xy[:, 1].long()]
            x = torch.cat([input_pts, input_xy], dim=-1)
        x = F.relu(self.input_layer(x))
        for i, layer in enumerate(self.hidden_layers):
            if i in self.skips:
                x = torch.cat([input_pts, x], -1)
            x = layer(x)
            if i == 4:
                x = self.bn1(x)
            x = F.relu(x)
        output = self.output_layer(x)
        if "epsilon_output" in self.config['model'] and self.config['model']['epsilon_output']:
            return self.config['model']['output_rate'] * output
        if "output_rate" in self.config['model']:
            return self.config['model']['output_rate'] * torch.tanh(output)
        return torch.tanh(output)