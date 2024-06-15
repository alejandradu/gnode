import torch
from torch import nn
from torch.nn import GRUCell, RNNCell

"""
All models must meet a few requirements
    1. They must have an init_model method that takes
    input_size and output_size as arguments
    2. They must have a forward method that takes inputs and hidden
    as arguments and returns output and hidden for one time step
    3. They must have a cell attribute that is the recurrent cell
    4. They must have a readout attribute that is the output layer
    (mapping from latent to output)
"""


class GRU_RNN(nn.Module):
    def __init__(
        self, latent_size, input_size=None, output_size=None, latent_ic_var=0.05
    ):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.cell = None
        self.readout = None
        self.latent_ics = torch.nn.Parameter(
            torch.zeros(latent_size), requires_grad=True
        )
        self.latent_ic_var = latent_ic_var

    def init_model(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.cell = GRUCell(input_size, self.latent_size)
        self.readout = nn.Linear(self.latent_size, output_size, bias=True)

    def init_hidden(self, batch_size):
        init_h = self.latent_ics.unsqueeze(0).expand(batch_size, -1)
        ic_noise = torch.randn_like(init_h) * self.latent_ic_var
        return init_h + ic_noise

    def forward(self, inputs, hidden):
        hidden = self.cell(inputs, hidden)
        output = self.readout(hidden)
        return output, hidden


class NoisyGRU(nn.Module):
    def __init__(
        self,
        latent_size,
        input_size=None,
        output_size=None,
        noise_level=0.05,
        latent_ic_var=0.05,
    ):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.cell = None
        self.readout = None
        self.noise_level = noise_level
        self.latent_ics = torch.nn.Parameter(
            torch.zeros(latent_size), requires_grad=True
        )
        self.latent_ic_var = latent_ic_var

    def init_hidden(self, batch_size):
        init_h = self.latent_ics.unsqueeze(0).expand(batch_size, -1)
        ic_noise = torch.randn_like(init_h) * self.latent_ic_var
        return init_h + ic_noise

    def init_model(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.cell = GRUCell(input_size, self.latent_size)
        self.readout = nn.Linear(self.latent_size, output_size, bias=True)

    def forward(self, inputs, hidden):
        hidden = self.cell(inputs, hidden)
        noise = torch.randn_like(hidden) * self.noise_level
        output = self.readout(hidden)
        hidden = hidden + noise
        return output, hidden


class NoisyGRU_RNN(nn.Module):
    def __init__(
        self,
        latent_size,
        input_size=None,
        output_size=None,
        noise_level=0.05,
        latent_ic_var=0.05,
    ):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.cell = None
        self.readout = None
        self.noise_level = noise_level
        self.latent_ics = torch.nn.Parameter(
            torch.zeros(latent_size), requires_grad=True
        )
        self.latent_ic_var = latent_ic_var

    def init_hidden(self, batch_size):
        init_h = self.latent_ics.unsqueeze(0).expand(batch_size, -1)
        ic_noise = torch.randn_like(init_h) * self.latent_ic_var
        return init_h + ic_noise

    def init_model(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.cell = GRUCell(input_size, self.latent_size)
        self.readout = nn.Linear(self.latent_size, output_size, bias=True)

    def forward(self, inputs, hidden):
        hidden = self.cell(inputs, hidden)
        noise = torch.randn_like(hidden) * self.noise_level
        output = self.readout(hidden)
        hidden = hidden + noise
        return output, hidden


class NoisyGRU_LatentL2(nn.Module):
    def __init__(
        self,
        latent_size,
        input_size=None,
        output_size=None,
        noise_level=0.05,
        latent_ic_var=0.05,
        l2_wt=1e-2,
    ):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.cell = None
        self.readout = None
        self.noise_level = noise_level
        self.l2_wt = l2_wt
        self.latent_ics = torch.nn.Parameter(
            torch.zeros(latent_size), requires_grad=True
        )
        self.latent_ic_var = latent_ic_var

    def init_hidden(self, batch_size):
        init_h = self.latent_ics.unsqueeze(0).expand(batch_size, -1)
        ic_noise = torch.randn_like(init_h) * self.latent_ic_var
        return init_h + ic_noise

    def init_model(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.cell = GRUCell(input_size, self.latent_size)
        self.readout = nn.Linear(self.latent_size, output_size, bias=True)

    def forward(self, inputs, hidden):
        hidden = self.cell(inputs, hidden)
        noise = torch.randn_like(hidden) * self.noise_level
        output = self.readout(hidden)
        hidden = hidden + noise
        return output, hidden

    def model_loss(self, loss_dict):
        latents = loss_dict["latents"]
        lats_flat = latents.view(latents.shape[0], -1)
        latent_l2_loss = self.l2_wt * torch.norm(lats_flat, p=2, dim=1).mean()
        return latent_l2_loss


class LintRNN(nn.Module):
    """Implement a vanilla RNN model following the equations used in 
    Valente et al. 2022 and as introduced by Sapolinsky et al."
    """
    def __init__(
        self,
        latent_size,
        input_size=None,
        output_size=None,
        noise_level=0.05,
        gamma=0.2,
        rank=128,
    ):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.readout = None
        self.noise_level = noise_level
        self.gamma = gamma
        self.act_func = nn.Tanh()
        self.rank=rank
        self.cell = None

    def init_model(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        if self.rank != self.latent_size:
            #self.recW2 = nn.Linear(self.latent_size, self.rank, bias=False)
            #self.recW1 = nn.Linear(self.rank, self.latent_size, bias=False)
            self.recW2 = nn.Parameter(torch.randn(self.latent_size, self.rank))
            self.recW1 = nn.Parameter(torch.randn(self.rank, self.latent_size))
        else:
            #self.recW = nn.Linear(self.latent_size, self.latent_size, bias=False)
            self.recW = nn.Parameter(torch.randn(self.latent_size, self.latent_size))
        #self.inpW = nn.Linear(self.input_size, self.latent_size, bias=False)
        self.inpW = nn.Parameter(torch.randn(self.input_size, self.latent_size))
        self.bias = nn.Parameter(torch.zeros(self.latent_size))
        self.readout = nn.Linear(self.latent_size, output_size, bias=True)
        # create a Torch class 

    def forward(self, inputs, hidden):
        noise = torch.randn_like(hidden) * self.noise_level
        output = self.readout(hidden)
        if self.rank != self.latent_size:
            hidden = (1 - self.gamma) * hidden + self.gamma * self.recW1.matmul(self.recW2.matmul(self.act_func(hidden))) + self.inpW.matmul(inputs) + self.bias + noise
        else:
            hidden = (1 - self.gamma) * hidden + self.gamma * self.act_func(hidden).matmul(self.recW) + inputs.matmul(self.inpW) + self.bias + noise
            # inputs/hidden/etc torch objects come as rows = batch size, columns = parameter size
        
        return output, hidden
    
    def clone_cell(self):
        new_net = LintRNN(self.latent_size, self.input_size, self.output_size, self.noise_level, self.gamma,
                          self.rank)
        return new_net
        
    
    
class DriscollRNN(nn.Module):
    def __init__(
        self,
        latent_size,
        input_size=None,
        output_size=None,
        noise_level=0.05,
        gamma=0.2,
    ):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.readout = None
        self.noise_level = noise_level
        self.gamma = gamma
        self.act_func = nn.Tanh()

    def init_model(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.recW = nn.Linear(self.latent_size, self.latent_size, bias=False)
        self.inpW = nn.Linear(self.input_size, self.latent_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(self.latent_size))
        self.readout = nn.Linear(self.latent_size, output_size, bias=True)

    def forward(self, inputs, hidden):
        noise = torch.randn_like(hidden) * self.noise_level
        output = self.readout(hidden)
        hidden = (1 - self.gamma) * self.recW(hidden) + self.gamma * self.act_func(
            self.recW(hidden) + self.inpW(inputs) + self.bias + noise
        )
        return output, hidden


class Vanilla_RNN(nn.Module):
    def __init__(self, latent_size, input_size=None, output_size=None):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.cell = None
        self.readout = None

    def init_model(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.cell = RNNCell(input_size, self.latent_size)
        self.readout = nn.Linear(self.latent_size, output_size)

    def forward(self, inputs, hidden=None):
        hidden = self.cell(inputs, hidden)
        output = self.readout(hidden)
        return output, hidden
    
    
# From Valente et al. 2022 (the "Lint" model)
class FullRankRNN(nn.Module):  

    def __init__(self, 
                 input_size, 
                 latent_size, 
                 output_size, 
                 noise_level, 
                 gamma=0.2, 
                 rho=1,
                 train_wi=False, 
                 train_wo=False, 
                 train_wrec=True, 
                 train_h0=False, 
                 train_si=True, 
                 train_so=True,
                 wi_init=None, 
                 wo_init=None, 
                 wrec_init=None, 
                 si_init=None, 
                 so_init=None, 
                 b_init=None,
                 add_biases=False, 
                 non_linearity=torch.tanh, 
                 output_non_linearity=torch.tanh):
        """
        :param input_size: int
        :param latent_size: int
        :param output_size: int
        :param noise_level: float
        :param gamma: float, value of dt/tau
        :param rho: float, std of gaussian distribution for initialization
        :param train_wi: bool
        :param train_wo: bool
        :param train_wrec: bool
        :param train_h0: bool
        :param wi_init: torch tensor of shape (input_dim, latent_size)
        :param wo_init: torch tensor of shape (latent_size, output_dim)
        :param wrec_init: torch tensor of shape (latent_size, latent_size)
        :param si_init: input scaling, torch tensor of shape (input_dim)
        :param so_init: output scaling, torch tensor of shape (output_dim)
        """
        super(FullRankRNN, self).__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.noise_level = noise_level
        self.gamma = gamma
        self.rho = rho
        self.train_wi = train_wi
        self.train_wo = train_wo
        self.train_wrec = train_wrec
        self.train_h0 = train_h0
        self.train_si = train_si
        self.train_so = train_so
        self.non_linearity = non_linearity
        self.output_non_linearity = output_non_linearity

        # Define parameters
        self.wi = nn.Parameter(torch.Tensor(input_size, latent_size))
        self.si = nn.Parameter(torch.Tensor(input_size))
        if train_wi:
            self.si.requires_grad = False
        else:
            self.wi.requires_grad = False
        if not train_si:
            self.si.requires_grad = False
        self.wrec = nn.Parameter(torch.Tensor(latent_size, latent_size))
        if not train_wrec:
            self.wrec.requires_grad = False
        self.b = nn.Parameter(torch.Tensor(latent_size))
        if not add_biases:
            self.b.requires_grad = False
        self.wo = nn.Parameter(torch.Tensor(latent_size, output_size))
        self.so = nn.Parameter(torch.Tensor(output_size))
        if train_wo:
            self.so.requires_grad = False
        if not train_wo:
            self.wo.requires_grad = False
        if not train_so:
            self.so.requires_grad = False
        self.h0 = nn.Parameter(torch.Tensor(latent_size))
        if not train_h0:
            self.h0.requires_grad = False

        # Initialize parameters
        with torch.no_grad():
            if wi_init is None:
                self.wi.normal_()
            else:
                self.wi.copy_(wi_init)
            if si_init is None:
                self.si.set_(torch.ones_like(self.si))
            else:
                self.si.copy_(si_init)
            if wrec_init is None:
                self.wrec.normal_(std=rho / sqrt(latent_size))
            else:
                self.wrec.copy_(wrec_init)
            if b_init is None:
                self.b.zero_()
            else:
                self.b.copy_(b_init)
            if wo_init is None:
                self.wo.normal_(std=1 / latent_size)
            else:
                self.wo.copy_(wo_init)
            if so_init is None:
                self.so.set_(torch.ones_like(self.so))
            else:
                self.so.copy_(so_init)
            self.h0.zero_()
        self.wi_full, self.wo_full = [None] * 2
        self._define_proxy_parameters()

    def _define_proxy_parameters(self):
        self.wi_full = (self.wi.t() * self.si).t()
        self.wo_full = self.wo * self.so

    def forward(self, input, return_dynamics=False, initial_states=None):
        """
        :param input: tensor of shape (batch_size, #timesteps, input_dimension)
        Important: the 3 dimensions need to be present, even if they are of size 1.
        :param return_dynamics: bool
        :param initial_states: None or torch tensor of shape (batch_size, latent_size) of initial state vectors for each trial if desired
        :return: if return_dynamics=False, output tensor of shape (batch_size, #timesteps, output_dimension)
                 if return_dynamics=True, (output tensor, trajectories tensor of shape (batch_size, #timesteps, #hidden_units))
        """
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        if initial_states is None:
            initial_states = self.h0
        h = initial_states.clone()
        r = self.non_linearity(initial_states)
        self._define_proxy_parameters()
        noise = torch.randn(batch_size, seq_len, self.latent_size, device=self.wrec.device)
        output = torch.zeros(batch_size, seq_len, self.output_size, device=self.wrec.device)
        if return_dynamics:
            trajectories = torch.zeros(batch_size, seq_len + 1, self.latent_size, device=self.wrec.device)
            trajectories[:, 0, :] = h

        # simulation loop
        for i in range(seq_len):
            h = h + self.noise_level * noise[:, i, :] + self.gamma * \
                (-h + r.matmul(self.wrec.t()) + input[:, i, :].matmul(self.wi_full))
            # r stands for reaodut
            r = self.non_linearity(h + self.b)
            # note that the output is the readout 
            output[:, i, :] = self.output_non_linearity(h) @ self.wo_full

            if return_dynamics:
                # the trajectories are just the latents
                trajectories[:, i + 1, :] = h

        if not return_dynamics:
            return output
        else:
            return output, trajectories

    def clone(self):
        # this returns the same module as in the cell attribute
        new_net = FullRankRNN(self.input_size, self.latent_size, self.output_size, self.noise_level, self.gamma,
                              self.rho, self.train_wi, self.train_wo, self.train_wrec, self.train_h0, self.train_si,
                              self.train_so, self.wi, self.wo, self.wrec, self.si, self.so, self.b, False,
                              self.non_linearity, self.output_non_linearity)
        return new_net
