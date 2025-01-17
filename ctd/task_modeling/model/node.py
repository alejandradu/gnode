import torch
from torch import nn


# TODO: Rename lowercase
class NODE(nn.Module):
    def __init__(
        self,
        num_layers,  # the length of MLP
        layer_hidden_size,  # the rank inside MLP 
        latent_size,  # dimension of h
        output_size=None,
        input_size=None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.layer_hidden_size = layer_hidden_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.input_size = input_size
        self.cell = None
        self.generator = None
        self.readout = None
        self.latent_ics = torch.nn.Parameter(
            torch.zeros(latent_size), requires_grad=True
        )

    def init_hidden(self, batch_size):
        return self.latent_ics.unsqueeze(0).expand(batch_size, -1)

    def init_model(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.cell = MLPCell(
            input_size, self.num_layers, self.layer_hidden_size, self.latent_size
        )
        self.generator = MLPCell(input_size, self.num_layers, self.layer_hidden_size, self.latent_size)
        self.readout = nn.Linear(self.latent_size, output_size)
        # Initialize weights and biases for the readout layer
        nn.init.normal_(
            self.readout.weight, mean=0.0, std=0.01
        )  # Small standard deviation
        nn.init.constant_(self.readout.bias, 0.0)  # Zero bias initialization

    def forward(self, inputs, hidden=None):
        n_samples, n_inputs = inputs.shape
        dev = inputs.device
        if hidden is None:
            hidden = torch.zeros((n_samples, self.latent_size), device=dev)
        hidden = self.generator(inputs, hidden)
        output = self.readout(hidden)
        return output, hidden


class MLPCell(nn.Module):
    def __init__(self, input_size, num_layers, layer_hidden_size, latent_size):
        super().__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.layer_hidden_size = layer_hidden_size
        self.latent_size = latent_size
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

    def forward(self, input, hidden):
        input_hidden = torch.cat([hidden, input], dim=1)
        return hidden + 0.1 * self.vf_net(input_hidden)
    

# add gating - CHECK THE EQUATIONS AND MAKE CORRECTIONS
class gNODE(nn.Module):
    def __init__(
        self,
        num_layers,
        layer_hidden_size,
        latent_size,
        output_size=None,
        input_size=None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.layer_hidden_size = layer_hidden_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.input_size = input_size
        self.cell = None
        self.generator = None
        self.readout = None
        self.latent_ics = torch.nn.Parameter(
            torch.zeros(latent_size), requires_grad=True
        )

    def init_hidden(self, batch_size):
        return self.latent_ics.unsqueeze(0).expand(batch_size, -1)

    def init_model(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.cell = MLPCell(
            input_size, self.num_layers, self.layer_hidden_size, self.latent_size
        )
        self.generator = MLPCell(input_size, self.num_layers, self.layer_hidden_size, self.latent_size)
        self.readout = nn.Linear(self.latent_size, output_size)
        self.gate = nn.Linear(self.latent_size, self.latent_size)
        # Initialize weights and biases for the readout layer
        nn.init.normal_(
            self.readout.weight, mean=0.0, std=0.01
        )  # Small standard deviation
        nn.init.constant_(self.readout.bias, 0.0)  # Zero bias initialization

    def forward(self, inputs, hidden=None):
            """
            Forward pass of the node model.

            Args:
                inputs (torch.Tensor): Input tensor of shape (n_samples, n_inputs).
                hidden (torch.Tensor, optional): Hidden state tensor of shape (n_samples, latent_size).
                    Defaults to None.

            Returns:
                output (torch.Tensor): Output tensor of shape (n_samples, output_size).
                hidden (torch.Tensor): Updated hidden state tensor of shape (n_samples, latent_size).
            """
            n_samples, n_inputs = inputs.shape
            dev = inputs.device
            if hidden is None:
                hidden = torch.zeros((n_samples, self.latent_size), device=dev)
            hidden = self.generator(inputs, hidden)
            gate = torch.sigmoid(self.gate(hidden))
            hidden = hidden * gate
            output = self.readout(hidden)
            return output, hidden
