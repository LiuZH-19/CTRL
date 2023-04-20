import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
import torchcde

# def noise_mask(seq_len, diff_dim, masking_ratio, lm=3, mode='separate', distribution='geometric', exclude_feats=None):
#     """
#     Creates a random boolean mask of the same shape as X, with 0s at places where a feature should be masked.
#     Args:
#         X: (seq_length, feat_dim) numpy array of features corresponding to a single sample
#         masking_ratio: proportion of seq_length to be masked. At each time step, will also be the proportion of
#             feat_dim that will be masked on average
#         lm: average length of masking subsequences (streaks of 0s). Used only when `distribution` is 'geometric'.
#         mode: whether each variable should be masked separately ('separate'), or all variables at a certain positions
#             should be masked concurrently ('concurrent')
#         distribution: whether each mask sequence element is sampled independently at random, or whether
#             sampling follows a markov chain (and thus is stateful), resulting in geometric distributions of
#             masked squences of a desired mean length `lm`
#         exclude_feats: iterable of indices corresponding to features to be excluded from masking (i.e. to remain all 1s)

#     Returns:
#         boolean numpy array with the same shape as X, with 0s at places where a feature should be masked
#     """
#     if exclude_feats is not None:
#         exclude_feats = set(exclude_feats)

#     if distribution == 'geometric':  # stateful (Markov chain)
#         if mode == 'separate':  # each variable (feature) is independent
#             mask = np.ones((seq_len, diff_dim), dtype=bool)
#             for m in range(diff_dim):  # feature dimension
#                 if exclude_feats is None or m not in exclude_feats:
#                     mask[:, m] = geom_noise_mask_single(seq_len, lm, masking_ratio)  # time dimension
#         else:  # replicate across feature dimension (mask all variables at the same positions concurrently)
#             mask = np.tile(np.expand_dims(geom_noise_mask_single(seq_len, lm, masking_ratio), 1), diff_dim)
#     else:  # each position is independent Bernoulli with p = 1 - masking_ratio
#         if mode == 'separate':
#             mask = np.random.choice(np.array([True, False]), size=(seq_len, diff_dim), replace=True,
#                                     p=(1 - masking_ratio, masking_ratio))
#         else:
#             mask = np.tile(np.random.choice(np.array([True, False]), size=(seq_len, 1), replace=True,
#                                             p=(1 - masking_ratio, masking_ratio)), diff_dim)

#     return mask

# def geom_noise_mask_single(L, lm, masking_ratio):
#     """
#     Randomly create a boolean mask of length `L`, consisting of subsequences of average length lm, masking with 0s a `masking_ratio`
#     proportion of the sequence L. The length of masking subsequences and intervals follow a geometric distribution.
#     Args:
#         L: length of mask and sequence to be masked
#         lm: average length of masking subsequences (streaks of 0s)
#         masking_ratio: proportion of L to be masked

#     Returns:
#         (L,) boolean numpy array intended to mask ('drop') with 0s a sequence of length L
#     """
#     keep_mask = np.ones(L, dtype=bool)
#     p_m = 1 / lm  # probability of each masking sequence stopping. parameter of geometric distribution.
#     p_u = p_m * masking_ratio / (1 - masking_ratio)  # probability of each unmasked sequence stopping. parameter of geometric distribution.
#     p = [p_m, p_u]

#     # Start in state 0 with masking_ratio probability
#     state = int(np.random.rand() > masking_ratio)  # state 0 means masking, 1 means not masking
#     for i in range(L):
#         keep_mask[i] = state  # here it happens that state and masking value corresponding to state are identical
#         if np.random.rand() < p[state]:
#             state = 1 - state

#     return keep_mask

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
        

        t = torch.linspace(0., seq_len-1, seq_len) #lzh 插值时用， 与后面积分的t无关
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
