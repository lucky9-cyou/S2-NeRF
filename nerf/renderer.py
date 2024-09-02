import math

import torch
import torch.nn as nn

import raymarching


class NeRFRenderer(nn.Module):

    def __init__(
        self,
        bound=1,
        density_scale=1,  # scale up deltas (or sigmas), to make the density grid more sharp. larger value than 1 usually improves performance.
        min_near=0.2,
        density_thresh=0.01,
        bg_radius=-1,
    ):
        super().__init__()

        self.bound = bound
        self.cascade = 1 + math.ceil(math.log2(bound))
        self.grid_size = 128
        self.density_scale = density_scale
        self.min_near = min_near
        self.density_thresh = density_thresh
        self.bg_radius = bg_radius  # radius of the background sphere.

        # prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
        # NOTE: aabb (can be rectangular) is only used to generate points, we still rely on bound (always cubic) to calculate density grid and hashing.
        aabb_train = torch.FloatTensor(
            [-bound, -bound, -bound, bound, bound, bound])
        aabb_infer = aabb_train.clone()
        self.register_buffer('aabb_train', aabb_train)
        self.register_buffer('aabb_infer', aabb_infer)

    def forward(self, x, d):
        raise NotImplementedError()

    # separated density and color query (can accelerate non-cuda-ray mode.)
    def density(self, x):
        raise NotImplementedError()

    def color(self, x, d, mask=None, **kwargs):
        raise NotImplementedError()

    def run(self,
            rays_o,
            rays_d,
            num_steps=128,
            upsample_steps=128,
            bg_color=None,
            perturb=False,
            **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # bg_color: [3] in range [0, 1]
        # return: image: [B, N, 3], depth: [B, N]

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0]  # N = B * N, in fact
        device = rays_o.device

        # choose aabb
        aabb = self.aabb_train if self.training else self.aabb_infer

        # sample steps
        nears, fars = raymarching.near_far_from_aabb_func(
            rays_o, rays_d, aabb, self.min_near)
        nears.unsqueeze_(-1)
        fars.unsqueeze_(-1)

        #print(f'nears = {nears.min().item()} ~ {nears.max().item()}, fars = {fars.min().item()} ~ {fars.max().item()}')

        z_vals = torch.linspace(0.0, 1.0, num_steps,
                                device=device).unsqueeze(0)  # [1, T]
        z_vals = z_vals.expand((N, num_steps))  # [N, T]
        z_vals = nears + (fars - nears) * z_vals  # [N, T], in [nears, fars]

        # perturb z_vals
        sample_dist = (fars - nears) / num_steps
        if perturb:
            z_vals = z_vals + (torch.rand(z_vals.shape, device=device) -
                               0.5) * sample_dist
            #z_vals = z_vals.clamp(nears, fars) # avoid out of bounds xyzs.

        # generate xyzs
        xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(
            -1)  # [N, 1, 3] * [N, T, 1] -> [N, T, 3]
        xyzs = torch.min(torch.max(xyzs, aabb[:3]), aabb[3:])  # a manual clip.

        #plot_pointcloud(xyzs.reshape(-1, 3).detach().cpu().numpy())

        # query SDF and RGB
        density_outputs = self.density(xyzs.reshape(-1, 3))

        #sigmas = density_outputs['sigma'].view(N, num_steps) # [N, T]
        for k, v in density_outputs.items():
            if v is None or 'inter' in k:
                continue
            density_outputs[k] = v.view(N, num_steps, -1)

        deltas = z_vals[..., 1:] - z_vals[..., :-1]  # [N, T+t-1]
        deltas = torch.cat(
            [deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)
        alphas = 1 - torch.exp(
            -deltas * self.density_scale *
            density_outputs['sigma'].squeeze(-1))  # [N, T+t]
        alphas_shifted = torch.cat(
            [torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15],
            dim=-1)  # [N, T+t+1]
        weights = alphas * torch.cumprod(alphas_shifted,
                                         dim=-1)[..., :-1]  # [N, T+t]

        if kwargs['add_dummy']:
            alphas_dummy = 1 - torch.exp(
                -deltas * self.density_scale *
                density_outputs['sigma_dummy'].squeeze(-1))  # [N, T+t]
            alphas_shifted_dummy = torch.cat([
                torch.ones_like(alphas_dummy[..., :1]),
                1 - alphas_dummy + 1e-15
            ],
                                             dim=-1)  # [N, T+t+1]
            weights_dummy = alphas_dummy * torch.cumprod(
                alphas_shifted_dummy, dim=-1)[..., :-1]  # [N, T+t]

        dirs = rays_d.view(-1, 1, 3).expand_as(xyzs)
        for k, v in density_outputs.items():
            if v is None or 'inter' in k:
                continue
            density_outputs[k] = v.view(-1, v.shape[-1])

        mask = weights > 1e-4  # hard coded
        rgbs_output = self.color(xyzs.reshape(-1, 3),
                                 dirs.reshape(-1, 3),
                                 mask=mask.reshape(-1),
                                 **density_outputs)
        rgbs = rgbs_output['rgbs'].view(N, -1, 3)  # [N, T+t, 3]

        if kwargs['add_dummy']:
            rgbs_dummy = rgbs_output['rgbs_dummy'].view(N, -1,
                                                        3)  # [N, T+t, 3]

        # calculate weight_sum (mask)
        weights_sum = weights.sum(dim=-1)  # [N]

        # calculate depth
        ori_z_vals = ((z_vals - nears) / (fars - nears)).clamp(0, 1)

        # absolute depth
        depth = torch.sum(weights * ori_z_vals, dim=-1)

        # calculate color
        image = torch.sum(weights.unsqueeze(-1) * rgbs,
                          dim=-2)  # [N, 3], in [0, 1]

        if kwargs['add_dummy']:
            weights_sum_dummy = weights_dummy.sum(dim=-1)  # [N]
            ori_z_vals_dummy = ((z_vals - nears) / (fars - nears)).clamp(0, 1)
            depth_dummy = torch.sum(weights_dummy * ori_z_vals_dummy, dim=-1)
            image_dummy = torch.sum(weights_dummy.unsqueeze(-1) * rgbs_dummy,
                                    dim=-2)

        # mix background color
        if self.bg_radius > 0:
            # use the bg model to calculate bg_color
            sph = raymarching.sph_from_ray(rays_o, rays_d,
                                           self.bg_radius)  # [N, 2] in [-1, 1]
            bg_color = self.background(sph, rays_d.reshape(-1, 3))  # [N, 3]
        elif bg_color is None:
            bg_color = 1

        image = image + (1 - weights_sum).unsqueeze(-1) * bg_color

        image = image.view(*prefix, 3)
        depth = depth.view(*prefix)

        if kwargs['add_dummy']:
            image_dummy = image_dummy + (
                1 - weights_sum_dummy).unsqueeze(-1) * bg_color
            image_dummy = image_dummy.view(*prefix, 3)
            depth_dummy = depth_dummy.view(*prefix)

        return {
            'depth':
            depth,
            'image':
            image,
            'weights_sum':
            weights_sum,
            'depth_dummy':
            depth_dummy if kwargs['add_dummy'] else None,
            'image_dummy':
            image_dummy if kwargs['add_dummy'] else None,
            'h_inter':
            density_outputs['h_inter']
            if 'h_inter' in density_outputs else None,
            'h_dummy_inter':
            density_outputs['h_dummy_inter']
            if 'h_dummy_inter' in density_outputs else None,
        }

    def render(self,
               rays_o,
               rays_d,
               staged=False,
               max_ray_batch=4096,
               **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: pred_rgb: [B, N, 3]

        B, N = rays_o.shape[:2]
        device = rays_o.device

        if staged:
            depth = torch.empty((B, N), device=device)
            if kwargs['add_dummy']:
                depth_dummy = torch.empty((B, N), device=device)
            image = torch.empty((B, N, 3), device=device)
            if kwargs['add_dummy']:
                image_dummy = torch.empty((B, N, 3), device=device)

            for b in range(B):
                head = 0
                while head < N:
                    tail = min(head + max_ray_batch, N)
                    results_ = self.run(rays_o[b:b + 1, head:tail],
                                        rays_d[b:b + 1, head:tail], **kwargs)
                    depth[b:b + 1, head:tail] = results_['depth']
                    if kwargs['add_dummy']:
                        depth_dummy[b:b + 1,
                                    head:tail] = results_['depth_dummy']
                    image[b:b + 1, head:tail] = results_['image']
                    if kwargs['add_dummy']:
                        image_dummy[b:b + 1,
                                    head:tail] = results_['image_dummy']
                    head += max_ray_batch

            results = {}
            results['depth'] = depth
            if kwargs['add_dummy']:
                results['depth_dummy'] = depth_dummy
            results['image'] = image
            if kwargs['add_dummy']:
                results['image_dummy'] = image_dummy
        else:
            results = self.run(rays_o, rays_d, **kwargs)

        return results
