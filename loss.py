import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class Contrastive_loss(nn.Module):
    def __init__(self, tau):
        super(Contrastive_loss, self).__init__()
        self.tau = tau

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1:torch.Tensor, z2:torch.Tensor):
        f = lambda x: torch.exp(x/self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def forward(self,z1:torch.Tensor, z2:torch.Tensor, mean:bool=True):
        l1 = self.semi_loss(z1, z2)
        l2 = self.semi_loss(z2, z1)
        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()
        return ret

def totalloss(hsi_lidar_logits, hsi_logits, tgt, lidar_logits, hsi_mu, hsi_logvar, lidar_mu, lidar_logvar, mu, logvar, z):
    hsi_kl_loss = -(1 + hsi_logvar - hsi_mu.pow(2) - hsi_logvar.exp()) / 2
    hsi_kl_loss = hsi_kl_loss.sum(dim=1).mean()

    lidar_kl_loss = -(1 + lidar_logvar - lidar_mu.pow(2) - lidar_logvar.exp()) / 2
    lidar_kl_loss = lidar_kl_loss.sum(dim=1).mean()

    kl_loss = -(1 + logvar - mu.pow(2) - logvar.exp()) / 2
    kl_loss = kl_loss.sum(dim=1).mean()
    IB_loss = F.cross_entropy(z, tgt)

    fusion_cls_loss = F.cross_entropy(hsi_lidar_logits, tgt)

    total_loss = fusion_cls_loss + 1e-3 * kl_loss + 1e-3 * hsi_kl_loss + 1e-3 * lidar_kl_loss + 1e-3 * IB_loss

    return total_loss

def KL_cross(mu_1, logvar_1, mu_2, logvar_2):
    var_1 = torch.exp(logvar_1)
    var_2 = torch.exp(logvar_2)
    KL_loss = logvar_2 - logvar_1 + ((var_1.pow(2) + (mu_1 - mu_2).pow(2)) / (2 * var_2.pow(2))) - 0.5
    KL_loss = KL_loss.sum(dim=1).mean()
    return KL_loss

def reparameterise(mu, std):
    """

    :param mu: [batch_size, z_dim]
    :param std: [batch_size, z_dim]
    :return:
    """
    # get epsilon from standard normal
    eps = torch.randn_like(std)
    return mu + std * eps


def con_loss(hsi_mu, hsi_logvar, lidar_mu, lidar_logvar):
    Conloss = Contrastive_loss(0.5)
    while True:
        h_z1 = reparameterise(hsi_mu, hsi_logvar)
        h_z2 = reparameterise(hsi_mu, hsi_logvar)

        if not np.array_equal(h_z1, h_z2):
            break
    while True:
        l_z1 = reparameterise(lidar_mu, lidar_logvar)
        l_z2 = reparameterise(lidar_mu, lidar_logvar)

        if not np.array_equal(l_z1, l_z2):
            break

    loss_h = Conloss(h_z1, h_z2)
    loss_l = Conloss(l_z1, l_z2)

    return loss_h + loss_l