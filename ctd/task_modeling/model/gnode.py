import torch
from torch import nn

# Implement a gated neural ordinary differential equation (gNODE)
# architecture for a neural network based on the implementation of
# a NODE in node.py


class gNODE(nn.Module):
    def __init__(
        self,
        dynamics_n_layers,    # the length of the MLP
        gating_n_layers,
        dynamics_hidden_size, # the rank inside MLP
        gating_hidden_size,
        latent_size,          # dimension of h
        gating_linear=False,
        output_size=None,  
        input_size=None,
        dt = 1,
        euler_step_size = 0.1
    ):
        super().__init__()
        self.dynamics_n_layers = dynamics_n_layers
        self.gating_n_layers = gating_n_layers
        self.dynamics_hidden_size = dynamics_hidden_size
        self.gating_hidden_size = gating_hidden_size
        self.latent_size = latent_size
        self.gating_linear = gating_linear
        self.output_size = output_size
        self.input_size = input_size
        self.dt = dt
        self.dynamics = None
        self.gating = None   # can be linear or an MLP - the MLP does not have to be complicated
        self.readout = None
        self.euler_step_size = euler_step_size
        self.latent_ics = torch.nn.Parameter(torch.zeros(latent_size), requires_grad=True)
        
    def init_hidden(self, batch_size):
        return self.latent_ics.unsqueeze(0).expand(batch_size, -1)

    def init_model(self, input_size, output_size):
        self.input_size = input_size 
        self.output_size = output_size
        
        if self.gating_linear:
            self.gating = nn.Linear(self.latent_size, self.latent_size)
        else:
            self.gating = MLPCell(self.input_size, self.gating_n_layers, self.gating_hidden_size, self.latent_size, self.euler_step_size)
        
        self.dynamics = MLPCell(input_size, self.dynamics_n_layers, self.dynamics_hidden_size, self.latent_size, self.euler_step_size)
        self.readout = nn.Linear(self.latent_size, self.output_size)
        # Initialize weights and biases for the readout layer
        nn.init.normal_(self.readout.weight, mean=0.0, std=0.01)  # Small standard deviation
        nn.init.constant_(self.readout.bias, 0.0)  # Zero bias initialization
        
    #@profile
    # TODO: revise this (it was wrong!!!)
    # want to implement another class entirely - do not use MLP
    def generator(self, inputs, hidden):
        if self.gating_linear:
            return self.dt * (self.gating(hidden) * (self.dynamics(inputs, hidden) - hidden)) + hidden
        else:
            return self.dt * (self.gating(inputs, hidden) * (self.dynamics(inputs, hidden) - hidden)) + hidden

    #@profile
    def forward(self, inputs, hidden=None):
        n_samples, n_inputs = inputs.shape
        dev = inputs.device
        if hidden is None:
            hidden = torch.zeros((n_samples, self.latent_size), device=dev)
        # this is like calling the ode solver
        hidden = self.generator(inputs, hidden)
        output = self.readout(hidden)
        return output, hidden


# this is just implementing the Euler method
# every call of MLPCell is like ode_solve()
# the output of each call is the solution to
# h(t+1) = h(t) + dt * f(h(t), x(t))
class MLPCell(nn.Module):
    def __init__(self, input_size, num_layers, layer_hidden_size, latent_size, euler_step_size=0.1):
        super().__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.layer_hidden_size = layer_hidden_size
        self.latent_size = latent_size
        self.euler_step_size = euler_step_size
        layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_size + latent_size, layer_hidden_size))
                layers.append(nn.ReLU())
            elif i == num_layers - 1:
                layers.append(nn.Linear(layer_hidden_size, latent_size))
            else:
                layers.append(nn.Linear(layer_hidden_size, layer_hidden_size))
                layers.append(nn.ReLU())
        self.vf_net = nn.Sequential(*layers)

    #@profile
    # implementing the Euler method as a forward pass
    def forward(self, input, hidden):
        # flattening all the paremeters that the function depends on
        # (the solver is agnostic to what they are, ir just solves)
        input_hidden = torch.cat([hidden, input], dim=1)
        # this is just an Euler update 
        return hidden + self.euler_step_size * self.vf_net(input_hidden)
    