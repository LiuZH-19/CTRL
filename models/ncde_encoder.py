import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
import torchcde

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("activation should be relu/gelu, not {}".format(activation))



class CDEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers):
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers

        self.linear_in = nn.Linear(hidden_channels, hidden_hidden_channels)
        
        self.linears = nn.ModuleList(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
                                           for _ in range(num_hidden_layers - 1))

        self.linear_out = nn.Linear(hidden_hidden_channels, input_channels * hidden_channels) 

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, t, z):
        z = self.linear_in(z)
        z = z.relu()


        for linear in self.linears:
            z = linear(z)
            z = z.relu()
        z = self.linear_out(z).view(*z.shape[:-1], self.hidden_channels, self.input_channels)  # #(batch_dims, hidden_channels, input_channels) 
        z = z.tanh()
        return z


######################
# Next, we need to package CDEFunc up into a model that computes the integral.
######################
class NCDEEncoder(torch.nn.Module):                 
    def __init__(self, input_dims, output_dims, intial_dims, hidden_dims=64, depth=4, mask_mode='geom', dropout=0.1, activation='gelu', interpolation="cubic", validating=False):
        super(NCDEEncoder, self).__init__()

        self.func = CDEFunc(input_dims+1, output_dims, hidden_dims , depth) 
        self.intial = nn.Sequential(
            nn.Linear(input_dims+1, intial_dims),  
            nn.ReLU(),
            nn.Linear(intial_dims, output_dims)
        )

        self.interpolation = interpolation
        self.mask_mode = mask_mode
        self.validating = validating

        self.act = _get_activation_fn(activation)
        self.dropout1 = nn.Dropout(dropout)
        self.output_layer = nn.Linear(output_dims, input_dims)
        

    def forward(self, x, mask=None):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """
        # x should be a tensor of shape (..., length, channels), and may have missing data represented by NaNs.

        b, seq_len, f = x.size()
  
        if mask is None:
            if self.training or self.validating:
                mask = self.mask_mode
            else:
                mask = 'all_true'
        
        if mask == 'geom':
            print('wrong: mask is None')       
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False
       
    
        x[~mask] = float('nan')
        

        t = torch.linspace(0., seq_len-1, seq_len) 
        t_ = t.unsqueeze(0).unsqueeze(-1).expand(b, seq_len, 1).to(x.device)
        x = torch.cat([t_, x], dim=-1) #(batch_size, seq_length, feat_dim+1)
        del t_
        ######################
        # Now we turn our dataset into a continuous path. We do this here via Hermite cubic spline interpolation.
        # The resulting `coeffs` is a tensor describing the path.
        ######################
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x)
        
        #natural_cubic_spline_coeffs

        if self.interpolation == 'cubic':
            X = torchcde.CubicSpline(coeffs)
        elif self.interpolation == 'linear':
            X = torchcde.LinearInterpolation(coeffs)
        else:
            raise ValueError("Only 'linear' and 'cubic' interpolation methods are implemented.")

        ######################
        # Easy to forget gotcha: Initial hidden state should be a function of the first observation.
        ######################
        X0 = X.evaluate(X.interval[0])
        z0 = self.intial(X0)


        z_T = torchcde.cdeint(X=X,
                              z0=z0,
                              func=self.func,
                              t=X.grid_points,
                              method='rk4',
                              options=dict(step_size=1)) #(batch_size, len(t), hidden_channels)       
        
        output = self.act(z_T)       
        output = self.dropout1(output)
        output = self.output_layer(output)  # (batch_size, seq_length, feat_dim)

        return z_T, output, ~mask
