# -*- coding : utf-8 -*-
# @Author : Xiao Han
# @Time   : 2025/5/21 20:52
import einops as ein
from einops.layers.torch import Rearrange
from tool import *
import math

class DNM_HX(nn.Module):
    def __init__(self, in_channel, out_channel, num_branch=2):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.W1 = nn.Parameter(torch.Tensor(out_channel, num_branch, in_channel))
        self.b1 = nn.Parameter(torch.Tensor(out_channel, num_branch))

        self.W2 = nn.Parameter(torch.Tensor(out_channel, 1, num_branch))
        self.b2 = nn.Parameter(torch.Tensor(out_channel, 1))

        self.reset_parameters()

    def reset_parameters(self):
        for o in range(self.out_channel):
            nn.init.kaiming_uniform_(self.W1[o], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.W2[o], a=math.sqrt(5))

            fan_in1, _ = nn.init._calculate_fan_in_and_fan_out(self.W1[o])
            bound1 = 1 / math.sqrt(fan_in1) if fan_in1 > 0 else 0
            nn.init.uniform_(self.b1[o], -bound1, bound1)

            fan_in2, _ = nn.init._calculate_fan_in_and_fan_out(self.W2[o])
            bound2 = 1 / math.sqrt(fan_in2) if fan_in2 > 0 else 0
            nn.init.uniform_(self.b2[o], -bound2, bound2)

    def forward(self, x, return_inter=False):
        if x.dim() == 2:
            x = x.unsqueeze(1)
            need_squeeze = True
        else:
            need_squeeze = False

        intermediate = torch.einsum('oni,bsi->bson', self.W1, x)
        intermediate += self.b1.unsqueeze(0).unsqueeze(0)

        if return_inter:
            return intermediate

        output = torch.einsum('odn,bson->bsod', self.W2, intermediate)
        output += self.b2.unsqueeze(0).unsqueeze(0)
        output = output.squeeze(-1)

        if need_squeeze:
            output = output.squeeze(1)

        return output



class ChannelMixer(nn.Module):
    def __init__(self, dim, num_patches, hidden_c, dropout=0.2):
        super(ChannelMixer, self).__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Conv1d(num_patches, hidden_c, kernel_size=3, padding=1)
        self.do1 = nn.Dropout(dropout)
        self.fc2 = nn.Conv1d(hidden_c, num_patches, kernel_size=1)
        self.do2 = nn.Dropout(dropout)
        self.act = F.gelu

    def forward(self, x):
        out = self.ln(x)
        out = self.fc1(out)
        out = self.do1(self.act(out))
        out = self.fc2(out)
        out = self.do2(out)
        return out + x

class TokenMixer(nn.Module):
    def __init__(self, dim, num_hidden, dropout=0.2):
        super(TokenMixer, self).__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = DNM_HX(dim, num_hidden)
        self.do1 = nn.Dropout(dropout)
        self.fc2 = DNM_HX(num_hidden, dim)
        self.do2 = nn.Dropout(dropout)
        self.act = F.gelu

    def forward(self, x):
        out = self.ln(x)
        out = self.do1(self.act(self.fc1(out)))
        out = self.do2(self.fc2(out))
        return out + x


class MixerLayer(nn.Module):
    def __init__(self, dim, num_patches, hidden_c, hidden_s, dropout):
        super(MixerLayer, self).__init__()
        self.channel2 = ChannelMixer(dim, num_patches, hidden_c, dropout)
        self.token = TokenMixer(dim, hidden_s, 0.2)

    def forward(self, x):
        x = self.channel2(x)
        x = self.token(x)
        return x


class Mixer(nn.Module):
    def __init__(self, img_size, patch_size, num_classes, dim, depth, hidden_c, hidden_s, is_cls_token, in_channels=3,
                 dropout=0.0, mlp_head='original'):
        super().__init__()

        assert img_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.num_patch = (img_size // patch_size) ** 2
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, dim, patch_size, patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )
        self.dim = dim
        self.is_cls_token = is_cls_token

        if self.is_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
            self.num_patch += 1

        self.mlp_blocks = nn.Sequential(
            *[
                MixerLayer(dim, self.num_patch, hidden_c, hidden_s, dropout)
                for _ in range(depth)
            ]
        )

        self.dnm = DNM_HX(dim, dim)
        if mlp_head == 'original':
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, num_classes)
            )
        elif mlp_head == 'None':
            self.mlp_head = nn.Identity()
        else:
            self.mlp_head = mlp_head

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        if self.is_cls_token:
            cls_tokens = ein.repeat(self.cls_token, '() n d -> b n d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)

        x = self.mlp_blocks(x)
        x = self.dnm(x)

        x = x[:, 0] if self.is_cls_token else x.mean(dim=1)
        return self.mlp_head(x)

class HSI_Clf(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(args.hsi_channel, args.in_channels, 1, 1),
            nn.BatchNorm2d(args.in_channels), nn.ReLU(),
            nn.Conv2d(args.in_channels, args.in_channels, 1, 1),
            nn.BatchNorm2d(args.in_channels), nn.ReLU()
        )

        self.Mixer = Mixer(args.img_size, args.patch_size, args.num_classes, args.dim, args.depth, args.hidden_c,
                           args.hidden_s, args.is_cls_token, args.in_channels, 0.0, args.mlp_head)
        self.mu = DNM_HX(args.dim, 128)
        self.logvar = DNM_HX(args.dim, 128)

        self.clf = DNM_HX(128, args.num_classes)

    def forward(self, x):
        x = self.proj(x)
        x = self.Mixer(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        x = self._reparameterize(mu, logvar)
        out = self.clf(x)
        return mu, logvar, out

    def _reparameterize(self, mu, logvar):
        std = torch.exp(logvar).sqrt()
        epsilon = torch.randn_like(std)
        sampler = epsilon * std
        return mu + sampler


class LiDAR_Clf(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(args.lidar_channel, args.in_channels, 1, 1),
            nn.BatchNorm2d(args.in_channels), nn.ReLU(),
            nn.Conv2d(args.in_channels, args.in_channels, 1, 1),
            nn.BatchNorm2d(args.in_channels), nn.ReLU(),
        )
        self.Mixer = Mixer(args.img_size, args.patch_size, args.num_classes, args.dim, args.depth, args.hidden_c,
                           args.hidden_s, args.is_cls_token, args.in_channels, 0.0, args.mlp_head)
        self.mu = DNM_HX(args.dim, 128)
        self.logvar = DNM_HX(args.dim, 128)

        self.clf = DNM_HX(128, args.num_classes)

    def forward(self, x):
        x = self.proj(x)
        x = self.Mixer(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        x = self._reparameterize(mu, logvar)
        out = self.clf(x)
        return mu, logvar, out

    def _reparameterize(self, mu, logvar):
        std = torch.exp(logvar).sqrt()
        epsilon = torch.randn_like(std)
        sampler = epsilon * std
        return mu + sampler


class UDM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.HSI_Clf = HSI_Clf(args)
        self.LiDAR_Clf = LiDAR_Clf(args)
        self.fusion = ChannelGate(3, 3, 'avg')
        self.mu = DNM_HX(128, 128)
        self.logvar = DNM_HX(128, 128)
        self.IB_classfier = DNM_HX(128, args.num_classes)
        self.fc_fusion1= DNM_HX(128, args.num_classes)

    def forward(self, x1, x2):
        hsi_mu, hsi_logvar, hsi_out = self.HSI_Clf(x1)
        lidar_mu, lidar_logvar, lidar_out = self.LiDAR_Clf(x2)

        hsi_var = torch.exp(hsi_logvar)
        lidar_var = torch.exp(lidar_logvar)

        def get_supp_mod(key):
            if key == "l":
                return lidar_mu
            elif key == "v":
                return hsi_mu
            else:
                raise KeyError

        l_sample, v_sample = cog_uncertainty_sample(hsi_mu, hsi_var, lidar_mu, lidar_logvar, sample_times=10)
        sample_dict = {
            "l": l_sample,  # hsi
            "v": v_sample  # lidar
        }
        cog_uncertainty_dict = {}
        with torch.no_grad():
            for key, sample_tensor in sample_dict.items():
                bsz, sample_times, dim = sample_tensor.shape
                sample_tensor = sample_tensor.reshape(bsz * sample_times, dim)
                sample_tensor = sample_tensor.unsqueeze(1)
                supp_mod = get_supp_mod(key)
                supp_mod = supp_mod.unsqueeze(1)
                supp_mod = supp_mod.unsqueeze(1).repeat(1, sample_times, 1, 1)
                supp_mod = supp_mod.reshape(bsz * sample_times, 1, dim)
                feature = torch.cat([supp_mod, sample_tensor], dim=1)

                feature_fusion = self.fusion(feature)
                mu = self.mu(feature_fusion)
                logvar = self.logvar(feature_fusion)
                z = reparameterise(mu, torch.exp(logvar))
                z = self.IB_classfier(z)
                hsi_lidar_out = self.fc_fusion1(mu)

                cog_un = torch.var(hsi_lidar_out, dim=-1)
                cog_uncertainty_dict[key] = cog_un

        cog_uncertainty_dict = cog_uncertainty_normal(cog_uncertainty_dict)

        weight = torch.softmax(torch.stack([lidar_var, hsi_var]), dim=0)
        lidar_w = weight[1]
        hsi_w = weight[0]

        feature_hsi = hsi_mu * hsi_w
        feature_lidar = lidar_mu * lidar_w

        feature = torch.stack((feature_hsi, feature_lidar), dim=1)
        feature_fusion = self.fusion(feature)
        mu = self.mu(feature_fusion)
        logvar = self.logvar(feature_fusion)
        z = reparameterise(mu, torch.exp(logvar))
        z = self.IB_classfier(z)
        hsi_lidar_out = self.fc_fusion1(mu)

        return [hsi_lidar_out, hsi_out, lidar_out, hsi_mu, hsi_logvar, lidar_mu, lidar_logvar, mu, logvar, z,
                cog_uncertainty_dict]