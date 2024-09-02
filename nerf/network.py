import torch
import torch.nn as nn
import torch.nn.functional as F

from activation import (add_gaussian_noise, add_mix_gaussian_noise,
                        add_partial_gaussian_noise, trunc_exp)
from encoding import get_encoder

from .renderer import NeRFRenderer


class NeRFNetwork(NeRFRenderer):

    def __init__(
        self,
        opt,
        encoding="hashgrid",
        encoding_dir="sphere_harmonics",
        encoding_bg="hashgrid",
        num_layers=2,
        hidden_dim=64,
        geo_feat_dim=15,
        num_layers_color=3,
        hidden_dim_color=64,
        num_layers_bg=2,
        hidden_dim_bg=64,
        bound=1,
        **kwargs,
    ):
        super().__init__(bound, **kwargs)

        self.opt = opt
        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.encoder, self.in_dim = get_encoder(encoding,
                                                desired_resolution=2048 *
                                                bound)

        if self.opt.add_dummy:
            if self.opt.dummy_layer < num_layers_color:
                raise ValueError(
                    'dummy layer should be larger than color layer')
            self.num_layers_dummy = self.opt.dummy_layer - num_layers_color

        sigma_net = []
        sigma_net_dummy = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
            else:
                in_dim = hidden_dim

            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim  # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim

            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

            if self.opt.add_dummy:
                if self.num_layers_dummy > 0 and l >= num_layers - self.num_layers_dummy:
                    sigma_net_dummy.append(
                        nn.Linear(in_dim, out_dim, bias=False))

        self.sigma_net = nn.ModuleList(sigma_net)
        if self.opt.add_dummy:
            self.sigma_net_dummy = nn.ModuleList(sigma_net_dummy)

        # color network
        self.num_layers_color = num_layers_color
        self.hidden_dim_color = hidden_dim_color
        self.encoder_dir, self.in_dim_dir = get_encoder(encoding_dir)
        if self.opt.add_dummy:
            self.encoder_dir_dummy, self.in_dim_dir_dummy = get_encoder(
                encoding_dir)

        color_net = []
        color_net_dummy = []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.in_dim_dir + self.geo_feat_dim
            else:
                in_dim = hidden_dim_color

            if l == num_layers_color - 1:
                out_dim = 3  # 3 rgb
            else:
                out_dim = hidden_dim_color

            color_net.append(nn.Linear(in_dim, out_dim, bias=False))

        # dummy network
        for l in range(self.opt.num_layers_color_dummy):
            if l == 0:
                in_dim = self.in_dim_dir + self.geo_feat_dim
            else:
                in_dim = hidden_dim_color

            if l == self.opt.num_layers_color_dummy - 1:
                out_dim = 3  # 3 rgb
            else:
                out_dim = hidden_dim_color

            if self.opt.add_dummy:
                color_net_dummy.append(nn.Linear(in_dim, out_dim, bias=False))

        self.color_net = nn.ModuleList(color_net)
        if self.opt.add_dummy:
            self.color_net_dummy = nn.ModuleList(color_net_dummy)

        # background network
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg
            self.hidden_dim_bg = hidden_dim_bg
            self.encoder_bg, self.in_dim_bg = get_encoder(
                encoding_bg,
                input_dim=2,
                num_levels=4,
                log2_hashmap_size=19,
                desired_resolution=2048)  # much smaller hashgrid

            bg_net = []
            for l in range(num_layers_bg):
                if l == 0:
                    in_dim = self.in_dim_bg + self.in_dim_dir
                else:
                    in_dim = hidden_dim_bg

                if l == num_layers_bg - 1:
                    out_dim = 3  # 3 rgb
                else:
                    out_dim = hidden_dim_bg

                bg_net.append(nn.Linear(in_dim, out_dim, bias=False))

            self.bg_net = nn.ModuleList(bg_net)
        else:
            self.bg_net = None

        if 'mix' == self.opt.noise_type:
            self.noise_func = add_mix_gaussian_noise
        elif 'random' == self.opt.noise_type:
            self.noise_func = add_gaussian_noise
        elif 'partial' == self.opt.noise_type:
            self.noise_func = add_partial_gaussian_noise

    def forward(self, x, d):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

        # sigma
        x = self.encoder(x, bound=self.bound)

        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        # color

        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)

        # sigmoid activation for rgb
        color = torch.sigmoid(h)

        return sigma, color

    def density(self, x):
        # x: [N, 3], in [-bound, bound]

        x = self.encoder(x, bound=self.bound)

        h = x
        first_layer = True
        for l in range(self.num_layers):

            # mlp -> noise -> mlp
            # mlp -> nosie -> dummy_mlp

            if self.opt.add_noise and l == self.num_layers - (
                    self.opt.noise_layer - self.num_layers_color):
                h = self.noise_func(h)

            if self.opt.add_dummy:
                if self.num_layers_dummy > 0 and l >= self.num_layers - self.num_layers_dummy:
                    if first_layer:
                        h_dummy, h_inter, first_layer = h.detach().clone(), h, False
                        h_dummy.requires_grad_()
                        h_dummy_inter = h_dummy
                    h_dummy = self.sigma_net_dummy[
                        l - (self.num_layers - self.num_layers_dummy)](h_dummy)
                    if l != self.num_layers - 1:
                        h_dummy = F.relu(h_dummy, inplace=True)

            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        if self.opt.add_dummy:
            if self.num_layers_dummy > 0:
                sigma_dummy = trunc_exp(h_dummy[..., 0])
                geo_feat_dummy = h_dummy[..., 1:]

        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
            'sigma_dummy': sigma_dummy if self.opt.add_dummy else None,
            'geo_feat_dummy': geo_feat_dummy if self.opt.add_dummy else None,
            'h_inter': h_inter if self.opt.add_dummy else None,
            'h_dummy_inter': h_dummy_inter if self.opt.add_dummy else None
        }

    def background(self, x, d):
        # x: [N, 2], in [-1, 1]

        h = self.encoder_bg(x)  # [N, C]
        d = self.encoder_dir(d)

        h = torch.cat([d, h], dim=-1)
        for l in range(self.num_layers_bg):
            h = self.bg_net[l](h)
            if l != self.num_layers_bg - 1:
                h = F.relu(h, inplace=True)

        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return rgbs

    # allow masked inference
    def color(self, x, d, mask=None, geo_feat=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.
        if self.opt.add_dummy:
            geo_feat_dummy = kwargs['geo_feat_dummy']

        if mask is not None:
            rgbs = torch.zeros(mask.shape[0],
                               3,
                               dtype=x.dtype,
                               device=x.device)  # [N, 3]

            if self.opt.add_dummy:
                rgbs_dummy = torch.zeros(mask.shape[0],
                                         3,
                                         dtype=x.dtype,
                                         device=x.device)

            # in case of empty mask
            if not mask.any():
                return {
                    'rgbs': rgbs,
                    'rgbs_dummy': rgbs_dummy if self.opt.add_dummy else None
                }
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]
            if self.opt.add_dummy:
                geo_feat_dummy = geo_feat_dummy[mask]

        _d = self.encoder_dir(d)
        h = torch.cat([_d, geo_feat], dim=-1)
        if self.opt.add_dummy:
            _d_dummy = self.encoder_dir_dummy(d)
            h_dummy = torch.cat([_d_dummy, geo_feat_dummy], dim=-1)

        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
    
        for l in range(self.opt.num_layers_color_dummy):
            if self.opt.add_dummy:
                h_dummy = self.color_net_dummy[l](h_dummy)
                if l != self.num_layers_color - 1:
                    h_dummy = F.relu(h_dummy, inplace=True)

        # sigmoid activation for rgb
        h = torch.sigmoid(h)
        if self.opt.add_dummy:
            h_dummy = torch.sigmoid(h_dummy)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype)  # fp16 --> fp32
        else:
            rgbs = h

        if self.opt.add_dummy:
            if mask is not None:
                rgbs_dummy[mask] = h_dummy.to(rgbs.dtype)
            else:
                rgbs_dummy = h_dummy

        return {
            'rgbs': rgbs,
            'rgbs_dummy': rgbs_dummy if self.opt.add_dummy else None
        }

    # optimizer utils
    def get_params(self, lr):

        params = [
            {
                'params': self.encoder.parameters(),
                'lr': lr
            },
            {
                'params': self.sigma_net.parameters(),
                'lr': lr
            },
            {
                'params': self.encoder_dir.parameters(),
                'lr': lr
            },
            {
                'params': self.color_net.parameters(),
                'lr': lr
            },
        ]

        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})

        return params

    def get_params_dummy(self, lr):

        params = [{
            'params': self.sigma_net_dummy.parameters(),
            'lr': lr
        }, {
            'params': self.color_net_dummy.parameters(),
            'lr': lr
        }, {
            'params': self.encoder_dir_dummy.parameters(),
            'lr': lr
        }]

        return params
