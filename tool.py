import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def reparameterise(mu, std):
    """
    mu : [batch_size,z_dim]
    std : [batch_size,z_dim]
    """
    # get epsilon from standard normal
    eps = torch.randn_like(std)
    return mu + std*eps

def cog_uncertainty_sample(mu_l, var_l, mu_v, var_v, sample_times=10):
    l_list = []
    for _ in range(sample_times):
        l_list.append(reparameterise(mu_l, var_l))
    l_sample = torch.stack(l_list, dim=1)

    v_list = []
    for _ in range(sample_times):
        v_list.append(reparameterise(mu_v, var_v))
    v_sample = torch.stack(v_list, dim=1)

    return l_sample, v_sample

def cog_uncertainty_normal(unc_dict, normal_type="None"):

    key_list = [k for k, _ in unc_dict.items()]
    comb_list = [t for _, t in unc_dict.items()]
    comb_t = torch.stack(comb_list, dim=1)
    mat = torch.exp(torch.reciprocal(comb_t))
    mat_sum = mat.sum(dim=-1, keepdim=True)
    weight = mat / mat_sum

    if normal_type == "minmax":
        weight = weight / torch.max(weight, dim=1)[0].unsqueeze(-1)  # [bsz, mod_num]
        for i, key in enumerate(key_list):
            unc_dict[key] = weight[:, i]
    else:
        pass
        # raise TypeError("Unsupported Operations at cog_uncertainty_normal!")

    return unc_dict

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super().__init__()
        self.gate_channels = gate_channels
        self.pool_types = pool_types

        self.con = nn.Conv1d(2, 2, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        channel_att_sum = None

        if self.pool_types == 'avg':
            avg_pool = x.mean(dim=2)
            avg_pool = avg_pool.unsqueeze(dim=2)
            channel_att_raw = self.con(avg_pool)

        elif self.pool_types == 'max':
            max_pool = x.max(dim=2)
            channel_att_raw = self.mlp(max_pool)
        elif self.pool_types == 'lp':
            # Calculate Lp pool along the feature_dim dimension
            lp_pool = x.norm(2, dim=2)
            channel_att_raw = self.mlp(lp_pool)
        elif self.pool_types == 'lse':
            # LSE pool only
            lse_pool = F.logsumexp_1d(x, dim=2)
            channel_att_raw = self.mlp(lse_pool)

        if channel_att_sum is None:
            channel_att_sum = channel_att_raw
        else:
            channel_att_sum = channel_att_sum + channel_att_raw

        score = avg_pool + channel_att_raw
        # x=x+x*scale
        x = x * score + x
        # fuison_output=torch.cat((x[:,0,:],x[:,1,:],x[:,2,:]),dim=1)
        fuison_output = (x[:, 0, :] + x[:, 1, :])

        return fuison_output