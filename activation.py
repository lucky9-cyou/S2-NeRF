import random

import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.nn.functional import dropout


class _trunc_exp(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)  # cast to float32
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(x.clamp(-15, 15))


trunc_exp = _trunc_exp.apply


class NoiseConfig:

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.gradient = None
        self.prob = 1.0
        self.std_scale = 1.0
        self.iter = 0
        self.total_iter = 0
        self.gradient_clip = False
        self.noise_decay = 1.0
        self.partial_scale = 10000


noise_config = NoiseConfig(0.0, 0.0)


class _add_gaussian_noise(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)  # cast to float32
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]

        g = torch.nan_to_num(g)
        if noise_config.gradient_clip:
            g = g.clamp(g.norm(dim=-1).median() * -1, g.norm(dim=-1).median())
            noise_std = g.norm(dim=-1).max(
            ) * noise_config.std * noise_config.std_scale * noise_config.noise_decay**min(
                noise_config.iter / noise_config.total_iter, 1)
        else:
            noise_std = g.norm(dim=-1).max(
            ) * noise_config.std * noise_config.std_scale * noise_config.noise_decay**min(
                noise_config.iter / noise_config.total_iter, 1)

        noise = torch.normal(noise_config.mean,
                             noise_std,
                             g.shape,
                             device=g.device)
        noise_config.gradient = g + noise
        noise_config.iter += 1
        return g + noise


add_gaussian_noise = _add_gaussian_noise.apply


class _add_partial_gaussian_noise(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)  # cast to float32
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        noise = torch.normal(noise_config.mean,
                             g.norm(dim=-1).max() * noise_config.std *
                             random.randint(100, 10000),
                             g.shape,
                             device=g.device)
        noise = noise * (torch.isclose(x, torch.tensor(0).float(),
                                       atol=0.0)).float()
        noise_config.gradient = g + noise
        return g + noise


add_partial_gaussian_noise = _add_partial_gaussian_noise.apply


class _add_mix_gaussian_noise(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)  # cast to float32
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]

        g = torch.nan_to_num(g)
        noise1 = torch.normal(noise_config.mean,
                              g.norm(dim=-1).max() * noise_config.std *
                              noise_config.partial_scale,
                              g.shape,
                              device=g.device)
        noise1 = noise1 * (torch.isclose(x, torch.tensor(0).float(),
                                         atol=0.0)).float()

        if noise_config.gradient_clip:
            g = g.clamp(g.norm(dim=-1).median() * -1, g.norm(dim=-1).median())
            noise_std = g.norm(dim=-1).max(
            ) * noise_config.std * noise_config.std_scale * noise_config.noise_decay**min(
                noise_config.iter / noise_config.total_iter, 1)
        else:
            noise_std = g.norm(dim=-1).max(
            ) * noise_config.std * noise_config.std_scale * noise_config.noise_decay**min(
                noise_config.iter / noise_config.total_iter, 1)

        noise2 = torch.normal(noise_config.mean,
                              noise_std,
                              g.shape,
                              device=g.device)

        noise2 = noise2 * (torch.logical_and(
            torch.logical_not(
                torch.isclose(x, torch.tensor(0).float(), atol=0.0)),
            (torch.rand(size=noise2.shape, device=noise2.device) <
             noise_config.prob))).float()

        noise_config.gradient = g + noise1 + noise2
        noise_config.iter += 1

        return g + noise1 + noise2


add_mix_gaussian_noise = _add_mix_gaussian_noise.apply
