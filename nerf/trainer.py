import glob
import os
import time

import cv2
import imageio
import matplotlib.cm as cm
import numpy as np
import tensorboardX
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import trimesh
from kornia.feature import LoFTR
from rich.console import Console
from torch_ema import ExponentialMovingAverage
from torchvision.transforms.functional import rgb_to_grayscale

from activation import noise_config

from .utils import (extract_geometry, linear_to_srgb, loftr_config,
                    make_matching_figure, srgb_to_linear)


class Trainer(object):

    def __init__(
        self,
        name,  # name of this experiment
        opt,  # extra conf
        model,  # network
        criterion=None,  # loss function, if None, assume inline implementation in train_step
        optimizer=None,  # optimizer
        optimizer_dummy=None,
        inerf_optimizer=None,
        inerf_optimizer_dummy=None,
        ema_decay=None,  # if use EMA, set the decay
        lr_scheduler=None,  # scheduler
        lr_scheduler_dummy=None,
        inerf_lr_scheduler=None,
        inerf_lr_scheduler_dummy=None,
        metrics=[],  # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
        metrics_dummy=[],  # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
        device=None,  # device to use, usually setting to None is OK. (auto choose device)
        mute=False,  # whether to mute all print
        fp16=False,  # amp optimize level
        eval_interval=1,  # eval once every $ epoch
        max_keep_ckpt=2,  # max num of saved ckpts in disk
        workspace='workspace',  # workspace to save logs & ckpts
        best_mode='min',  # the smaller/larger result, the better
        use_loss_as_metric=True,  # use loss as the first metric
        report_metric_at_train=False,  # also report metrics at training
        use_checkpoint="latest",  # which ckpt to use at init time
        use_tensorboardX=True,  # whether to use tensorboard for logging
        scheduler_update_every_step=False,  # whether to call scheduler.step() after every train step
        train_dataset=None,
    ):

        self.name = name
        self.opt = opt
        self.mute = mute
        self.metrics = metrics
        self.metrics_dummy = metrics_dummy
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.train_dataset = train_dataset
        self.console = Console()
        self.device = device

        if self.opt.inerf:
            self.inerf_feature_matching_config = loftr_config
            self.inerf_feature_matching_config['match_coarse'][
                'thr'] = self.opt.inerf_thr
            self.matcher = LoFTR(
                "indoor_new",
                self.inerf_feature_matching_config).cuda().train()

        model.to(self.device)
        self.model = model

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        import lpips
        self.criterion_lpips = lpips.LPIPS(net='alex').to(self.device)

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=0.001,
                                        weight_decay=5e-4)  # naive adam
        else:
            self.optimizer = optimizer(self.model)
            if self.opt.add_dummy and not self.opt.inerf:
                self.optimizer_dummy = optimizer_dummy(self.model)
            if self.opt.inerf:
                self.inerf_optimizer = inerf_optimizer(self.model)
                self.inerf_optimizer_dummy = inerf_optimizer_dummy(self.model)

                self.optimizer_dummy = optim.Adam(
                    self.model.parameters(), lr=0.001,
                    weight_decay=5e-4)  # naive adam

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda epoch: 1)  # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)
            if self.opt.add_dummy and not self.opt.inerf:
                self.lr_scheduler_dummy = lr_scheduler_dummy(
                    self.optimizer_dummy)

            if self.opt.inerf:
                self.inerf_lr_scheduler = inerf_lr_scheduler(
                    self.inerf_optimizer)
                self.inerf_lr_scheduler_dummy = inerf_lr_scheduler_dummy(
                    self.inerf_optimizer_dummy)

                self.lr_scheduler_dummy = optim.lr_scheduler.LambdaLR(
                    self.optimizer_dummy,
                    lr_lambda=lambda epoch: 1)  # fake scheduler

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(),
                                                decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.inerf_global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [],  # metrics[0], or valid_loss
            "checkpoints":
            [],  # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }
        self.grad_label_ratio = 0.0

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)

        self.log(
            f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}'
        )
        self.log(
            f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}'
        )

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(
                        f"[INFO] {self.best_path} not found, loading latest ..."
                    )
                    self.load_checkpoint()
            else:  # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)

    def __del__(self):
        if self.log_ptr:
            self.log_ptr.close()

    def log(self, *args, **kwargs):
        if not self.mute:
            #print(*args)
            self.console.print(*args, **kwargs)
        if self.log_ptr:
            print(*args, file=self.log_ptr)
            self.log_ptr.flush()  # write immediately to file

    ### ------------------------------

    def train_step(self, data):

        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]

        images = data['images']  # [B, N, 3/4]
        if self.opt.add_dummy:
            images_dummy = data['images_dummy']  # [B, N, 3/4]

        B, N, C = images.shape

        if self.opt.color_space == 'linear':
            images[..., :3] = srgb_to_linear(images[..., :3])

        if C == 3 or self.model.bg_radius > 0:
            bg_color = 1
        # train with random background color if not using a bg model and has alpha channel.
        else:
            #bg_color = torch.ones(3, device=self.device) # [3], fixed white background
            #bg_color = torch.rand(3, device=self.device) # [3], frame-wise random.
            bg_color = torch.rand_like(
                images[..., :3])  # [N, 3], pixel-wise random.

        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (
                1 - images[..., 3:])
            if self.opt.add_dummy:
                gt_rgb_dummy = images_dummy[..., :3] * images_dummy[
                    ..., 3:] + bg_color * (1 - images_dummy[..., 3:])
        else:
            gt_rgb = images
            if self.opt.add_dummy:
                gt_rgb_dummy = images_dummy

        outputs = self.model.render(rays_o,
                                    rays_d,
                                    staged=False,
                                    bg_color=bg_color,
                                    perturb=True,
                                    force_all_rays=False,
                                    **vars(self.opt))

        pred_rgb = outputs['image']
        if self.opt.add_dummy:
            pred_rgb_dummy = outputs['image_dummy']

        # MSE loss
        loss = self.criterion(pred_rgb,
                              gt_rgb).mean(-1)  # [B, N, 3] --> [B, N]
        if self.opt.add_dummy:
            loss_dummy = self.criterion(pred_rgb_dummy, gt_rgb_dummy).mean(-1)

        loss = loss.mean()
        loss_dummy = loss_dummy.mean() if self.opt.add_dummy else None

        if self.opt.add_dummy:
            if not self.opt.add_noise:
                grad_inter = torch.autograd.grad(
                    loss,
                    outputs['h_inter'],
                    grad_outputs=torch.ones_like(loss),
                    create_graph=True)[0]
            grad_inter_dummy = torch.autograd.grad(
                loss_dummy,
                outputs['h_dummy_inter'],
                grad_outputs=torch.ones_like(loss_dummy),
                create_graph=True)[0]

        if self.opt.add_dummy:
            if self.opt.add_noise:
                return pred_rgb, pred_rgb_dummy, gt_rgb, gt_rgb_dummy, loss, loss_dummy, grad_inter_dummy, None
            else:
                return pred_rgb, pred_rgb_dummy, gt_rgb, gt_rgb_dummy, loss, loss_dummy, grad_inter_dummy, grad_inter
        return pred_rgb, gt_rgb, loss

    def inerf_train_step(self, data):

        inerf_rays_o = data['inerf_rays_o']  # [B, N, 3]
        inerf_rays_d = data['inerf_rays_d']  # [B, N, 3]

        images = data['images']  # [B, H, W, 3/4]
        depths = data['depths'].squeeze(2)  # [B, H, W, 1]

        B, H, W, C = images.shape

        if self.opt.color_space == 'linear':
            images[..., :3] = srgb_to_linear(images[..., :3])

        if C == 3 or self.model.bg_radius > 0:
            bg_color = 1
        # train with random background color if not using a bg model and has alpha channel.
        else:
            #bg_color = torch.ones(3, device=self.device) # [3], fixed white background
            #bg_color = torch.rand(3, device=self.device) # [3], frame-wise random.
            bg_color = torch.rand_like(
                images[..., :3])  # [N, 3], pixel-wise random.

        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (
                1 - images[..., 3:])
        else:
            gt_rgb = images

        inerf_outputs = self.model.render(inerf_rays_o,
                                          inerf_rays_d,
                                          staged=False,
                                          bg_color=bg_color,
                                          perturb=True,
                                          force_all_rays=False,
                                          **vars(self.opt))

        pred_depth_dummy = inerf_outputs['depth_dummy']
        pred_image_dummy = inerf_outputs['image_dummy']
        pred_image = images.reshape(B, -1, C)

        # MSE loss
        if self.inerf_global_step <= self.opt.inerf_train_steps:
            pred_depth = F.normalize(torch.nan_to_num(depths.reshape(B, -1)),
                                     1)
            pred_depth_dummy = F.normalize(torch.nan_to_num(pred_depth_dummy),
                                           1)
            loss = self.criterion(pred_depth, pred_depth_dummy).mean(-1)
        else:
            pred_image = torch.nan_to_num(pred_image)
            pred_image_dummy = torch.nan_to_num(pred_image_dummy)
            loss = self.criterion(pred_image, pred_image_dummy).mean(-1)

        loss = loss.mean()

        return pred_image, pred_image_dummy, loss

    def eval_step(self, data):

        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]
        images = data['images']  # [B, H, W, 3/4]
        B, H, W, C = images.shape

        if self.opt.color_space == 'linear':
            images[..., :3] = srgb_to_linear(images[..., :3])

        # eval with fixed background color
        bg_color = 1
        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (
                1 - images[..., 3:])
        else:
            gt_rgb = images

        outputs = self.model.render(rays_o,
                                    rays_d,
                                    staged=True,
                                    bg_color=bg_color,
                                    perturb=False,
                                    **vars(self.opt))

        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W)
        loss = self.criterion(pred_rgb, gt_rgb).mean()

        if self.opt.add_dummy:
            pred_rgb_dummy = outputs['image_dummy'].reshape(B, H, W, 3)
            pred_depth_dummy = outputs['depth_dummy'].reshape(B, H, W)
            loss_dummy = self.criterion(pred_rgb_dummy, gt_rgb).mean()
            return pred_rgb, pred_rgb_dummy, pred_depth, pred_depth_dummy, gt_rgb, loss, loss_dummy

        return pred_rgb, pred_depth, gt_rgb, loss

    def inerf_eval_step(self, data):

        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]
        inerf_rays_o = data['inerf_rays_o']  # [B, N, 3]
        inerf_rays_d = data['inerf_rays_d']  # [B, N, 3]
        images = data['images']  # [B, H, W, 3/4]
        B, H, W, C = images.shape

        if self.opt.color_space == 'linear':
            images[..., :3] = srgb_to_linear(images[..., :3])

        # eval with fixed background color
        bg_color = 1
        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (
                1 - images[..., 3:])
        else:
            gt_rgb = images

        outputs = self.model.render(rays_o,
                                    rays_d,
                                    staged=True,
                                    bg_color=bg_color,
                                    perturb=False,
                                    **vars(self.opt))

        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W)
        loss = self.criterion(pred_rgb, gt_rgb).mean()

        inerf_outputs = self.model.render(inerf_rays_o,
                                          inerf_rays_d,
                                          staged=True,
                                          bg_color=bg_color,
                                          perturb=False,
                                          **vars(self.opt))
        pred_rgb_dummy = inerf_outputs['image_dummy'].reshape(B, H, W, 3)
        pred_depth_dummy = inerf_outputs['depth_dummy'].reshape(B, H, W)

        loss_dummy = self.criterion(pred_rgb_dummy, pred_rgb).mean()
        return pred_rgb, pred_rgb_dummy, pred_depth, pred_depth_dummy, gt_rgb, loss, loss_dummy

    # moved out bg_color and perturb for more flexible control...
    def test_step(self, data, bg_color=None, perturb=False):

        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]
        H, W = data['H'], data['W']

        if bg_color is not None:
            bg_color = bg_color.to(self.device)

        outputs = self.model.render(rays_o,
                                    rays_d,
                                    staged=True,
                                    bg_color=bg_color,
                                    perturb=perturb,
                                    **vars(self.opt))

        pred_rgb = outputs['image'].reshape(-1, H, W, 3)
        pred_depth = outputs['depth'].reshape(-1, H, W)

        if self.opt.add_dummy:
            pred_rgb_dummy = outputs['image_dummy'].reshape(-1, H, W, 3)
            pred_depth_dummy = outputs['depth_dummy'].reshape(-1, H, W)
            return pred_rgb, pred_rgb_dummy, pred_depth, pred_depth_dummy

        return pred_rgb, pred_depth

    def save_mesh(self, save_path=None, resolution=256, threshold=10):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'meshes',
                                     f'{self.name}_{self.epoch}.ply')

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        def query_func(pts):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    sigma = self.model.density(pts.to(self.device))['sigma']
            return sigma

        vertices, triangles = extract_geometry(self.model.aabb_infer[:3],
                                               self.model.aabb_infer[3:],
                                               resolution=resolution,
                                               threshold=threshold,
                                               query_func=query_func)

        mesh = trimesh.Trimesh(
            vertices, triangles,
            process=False)  # important, process=True leads to seg fault...
        mesh.export(save_path)

        self.log(f"==> Finished saving mesh.")

    ### ------------------------------

    def train(self,
              train_loader,
              valid_loader,
              max_epochs,
              train_dataset=None):
        if self.use_tensorboardX:
            self.writer = tensorboardX.SummaryWriter(
                os.path.join(self.workspace, "run", self.name))

        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch

            self.train_one_epoch(train_loader, train_dataset)

            if self.workspace is not None:
                self.save_checkpoint(full=True,
                                     best=False,
                                     train_dataset=train_dataset)

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader,
                                        train_dataset=train_dataset)
                self.save_checkpoint(full=False,
                                     best=True,
                                     train_dataset=train_dataset)

        if self.use_tensorboardX:
            self.writer.close()

    def inerf_train(self, train_loader, valid_loader, max_epochs,
                    train_dataset, val_dataset):
        if self.use_tensorboardX:
            self.writer = tensorboardX.SummaryWriter(
                os.path.join(self.workspace, "run", self.name))

        for epoch in range(self.epoch + 1, self.epoch + max_epochs + 1):
            self.epoch = epoch

            self.inerf_one_epoch(train_loader, train_dataset)

            if self.workspace is not None:
                self.save_checkpoint(full=True,
                                     best=False,
                                     train_dataset=train_dataset)

            if self.epoch % self.eval_interval == 0:
                val_dataset.camera_poses = train_dataset.camera_poses
                val_dataset.inerf_poses = train_dataset.inerf_poses
                self.inerf_evaluate_one_epoch(valid_loader,
                                              train_dataset=train_dataset)
                self.save_checkpoint(full=False,
                                     best=True,
                                     train_dataset=train_dataset)

        if self.use_tensorboardX:
            self.writer.close()

    def evaluate(self, loader, name=None):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader, name)
        self.use_tensorboardX = use_tensorboardX

    def test(self, loader, save_path=None, name=None, write_video=True):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'results')

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        os.makedirs(save_path, exist_ok=True)

        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(
            total=len(loader) * loader.batch_size,
            bar_format=
            '{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        self.model.eval()

        if write_video:
            all_preds = []
            all_preds_depth = []

            if self.opt.add_dummy:
                all_preds_dummy = []
                all_preds_depth_dummy = []

        with torch.no_grad():

            for i, data in enumerate(loader):

                with torch.cuda.amp.autocast(enabled=self.fp16):

                    if self.opt.add_dummy:
                        preds, preds_dummy, preds_depth, preds_depth_dummy = self.test_step(
                            data)
                    else:
                        preds, preds_depth = self.test_step(data)

                if self.opt.color_space == 'linear':
                    preds = linear_to_srgb(preds)
                    if self.opt.add_dummy:
                        preds_dummy = linear_to_srgb(preds_dummy)

                pred = preds[0].detach().cpu().numpy()
                pred = (pred * 255).astype(np.uint8)

                pred_depth = preds_depth[0].detach().cpu().numpy()
                pred_depth = (pred_depth * 255).astype(np.uint8)

                if self.opt.add_dummy:
                    pred_dummy = preds_dummy[0].detach().cpu().numpy()
                    pred_dummy = (pred_dummy * 255).astype(np.uint8)

                    pred_depth_dummy = preds_depth_dummy[0].detach().cpu(
                    ).numpy()
                    pred_depth_dummy = (pred_depth_dummy * 255).astype(
                        np.uint8)

                if write_video:
                    all_preds.append(pred)
                    all_preds_depth.append(pred_depth)

                    if self.opt.add_dummy:
                        all_preds_dummy.append(pred_dummy)
                        all_preds_depth_dummy.append(pred_depth_dummy)
                else:
                    cv2.imwrite(
                        os.path.join(save_path, f'{name}_{i:04d}_rgb.png'),
                        cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(
                        os.path.join(save_path, f'{name}_{i:04d}_depth.png'),
                        pred_depth)

                    if self.opt.add_dummy:
                        cv2.imwrite(
                            os.path.join(save_path,
                                         f'{name}_{i:04d}_rgb_dummy.png'),
                            cv2.cvtColor(pred_dummy, cv2.COLOR_RGB2BGR))
                        cv2.imwrite(
                            os.path.join(save_path,
                                         f'{name}_{i:04d}_depth_dummy.png'),
                            pred_depth_dummy)

                pbar.update(loader.batch_size)

        if write_video:
            all_preds = np.stack(all_preds, axis=0)
            all_preds_depth = np.stack(all_preds_depth, axis=0)
            imageio.mimwrite(os.path.join(save_path, f'{name}_rgb.mp4'),
                             all_preds,
                             fps=25,
                             quality=8,
                             macro_block_size=1)
            imageio.mimwrite(os.path.join(save_path, f'{name}_depth.mp4'),
                             all_preds_depth,
                             fps=25,
                             quality=8,
                             macro_block_size=1)

            if self.opt.add_dummy:
                all_preds_dummy = np.stack(all_preds_dummy, axis=0)
                all_preds_depth_dummy = np.stack(all_preds_depth_dummy, axis=0)
                imageio.mimwrite(os.path.join(save_path,
                                              f'{name}_rgb_dummy.mp4'),
                                 all_preds_dummy,
                                 fps=25,
                                 quality=8,
                                 macro_block_size=1)
                imageio.mimwrite(os.path.join(save_path,
                                              f'{name}_depth_dummy.mp4'),
                                 all_preds_depth_dummy,
                                 fps=25,
                                 quality=8,
                                 macro_block_size=1)

        self.log(f"==> Finished Test.")

    def train_one_epoch(self, loader, train_dataset=None):
        self.log(
            f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ..."
        )

        total_loss = 0
        if self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        pbar = tqdm.tqdm(
            total=len(loader) * loader.batch_size,
            bar_format=
            '{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )

        self.local_step = 0

        for data in loader:

            self.local_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()
            if self.opt.add_dummy:
                self.optimizer_dummy.zero_grad()

            if self.opt.add_dummy:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_dummy, truths, truths_dummy, loss, loss_dummy, grad_intern_dummy, grad_intern = self.train_step(
                        data)
            else:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, truths, loss = self.train_step(data)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)

            if self.opt.add_dummy:
                if self.opt.add_noise:
                    loss_grad = F.mse_loss(noise_config.gradient.detach(),
                                           grad_intern_dummy,
                                           reduction='mean')
                else:
                    loss_grad = F.mse_loss(grad_intern.detach(),
                                           grad_intern_dummy,
                                           reduction='mean')

                if self.global_step <= 100:
                    self.grad_label_ratio += loss_dummy.detach(
                    ) / loss_grad.detach()
                    grad_label_ratio = self.grad_label_ratio / self.global_step
                else:
                    grad_label_ratio = self.grad_label_ratio / 100

                if self.opt.lambda_grad > 1e5:
                    loss_grad *= self.opt.lambda_grad
                else:
                    loss_grad *= grad_label_ratio * self.opt.lambda_grad

            if self.opt.add_dummy:
                if self.opt.lambda_grad != 0.0 and self.opt.lambda_label == 0.0:
                    self.scaler.scale(loss_grad).backward()
                if self.opt.lambda_grad != 0.0 and self.opt.lambda_label != 0.0:
                    self.scaler.scale(loss_grad).backward(retain_graph=True)
                if self.opt.lambda_label != 0.0:
                    self.scaler.scale(loss_dummy).backward()
                self.scaler.step(self.optimizer_dummy)

            self.scaler.update()

            if train_dataset is not None and self.opt.add_dummy:
                train_dataset.images_dummy.data = train_dataset.images_dummy.data.clamp(
                    0, 1)

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

                if self.opt.add_dummy:
                    self.lr_scheduler_dummy.step()

            loss_val = loss.item()
            loss_dummy_val = loss_dummy.item() if self.opt.add_dummy else 0
            loss_grad_val = loss_grad.item() if self.opt.add_dummy else 0
            if self.opt.add_dummy:
                del loss_dummy, loss_grad, grad_intern_dummy, grad_intern
            total_loss += loss_val

            if self.report_metric_at_train:
                for metric in self.metrics:
                    metric.update(preds, truths)

            if self.use_tensorboardX:
                self.writer.add_scalar("train/loss", loss_val,
                                       self.global_step)
                self.writer.add_scalar("train/lr",
                                       self.optimizer.param_groups[0]['lr'],
                                       self.global_step)

                if self.opt.add_dummy:
                    self.writer.add_scalar("train/loss_dummy", loss_dummy_val,
                                           self.global_step)
                    self.writer.add_scalar("train/loss_grad", loss_grad_val,
                                           self.global_step)

            if self.scheduler_update_every_step:
                pbar.set_description(
                    f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}"
                )
            else:
                pbar.set_description(
                    f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
            pbar.update(loader.batch_size)

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        pbar.close()
        if self.report_metric_at_train:
            for metric in self.metrics:
                self.log(metric.report(), style="red")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="train")
                metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler,
                          torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")

    def evaluate_one_epoch(self, loader, name=None, train_dataset=None):
        self.log(f"++> Evaluate at epoch {self.epoch} ...")

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        total_loss = 0
        for metric in self.metrics:
            metric.clear()

        if self.opt.add_dummy:
            for metric in self.metrics_dummy:
                metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        pbar = tqdm.tqdm(
            total=len(loader) * loader.batch_size,
            bar_format=
            '{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )

        with torch.no_grad():
            self.local_step = 0

            for data in loader:
                self.local_step += 1

                if self.opt.add_dummy:
                    with torch.cuda.amp.autocast(enabled=self.fp16):
                        preds, preds_dummy, preds_depth, pred_depth_dummy, truths, loss, loss_dummy = self.eval_step(
                            data)
                else:
                    with torch.cuda.amp.autocast(enabled=self.fp16):
                        preds, preds_depth, truths, loss = self.eval_step(data)

                loss_val = loss.item()
                total_loss += loss_val

                for metric in self.metrics:
                    metric.update(preds, truths)

                if self.opt.add_dummy:
                    for metric in self.metrics_dummy:
                        metric.update(preds_dummy, truths)

                # save image
                save_path = os.path.join(
                    self.workspace, 'validation',
                    f'{name}_{self.local_step:04d}_rgb.png')
                save_path_depth = os.path.join(
                    self.workspace, 'validation',
                    f'{name}_{self.local_step:04d}_depth.png')

                if self.opt.add_dummy:
                    save_path_dummy = os.path.join(
                        self.workspace, 'validation',
                        f'{name}_{self.local_step:04d}_rgb_dummy.png')
                    save_path_depth_dummy = os.path.join(
                        self.workspace, 'validation',
                        f'{name}_{self.local_step:04d}_depth_dummy.png')

                if train_dataset is not None:
                    save_path_train = os.path.join(
                        self.workspace, 'validation',
                        f'{name}_{self.local_step:04d}_rgb_train.png')

                #self.log(f"==> Saving validation image to {save_path}")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                if self.opt.color_space == 'linear':
                    preds = linear_to_srgb(preds)

                pred = preds[0].detach().cpu().numpy()
                pred = (pred * 255).astype(np.uint8)

                pred_depth = preds_depth[0].detach().cpu().numpy()
                pred_depth = (pred_depth * 255).astype(np.uint8)

                cv2.imwrite(save_path, cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                cv2.imwrite(save_path_depth, pred_depth)

                if self.opt.add_dummy:
                    pred_dummy = preds_dummy[0].detach().cpu().numpy()
                    pred_dummy = (pred_dummy * 255).astype(np.uint8)

                    pred_depth_dummy = pred_depth_dummy[0].detach().cpu(
                    ).numpy()
                    pred_depth_dummy = (pred_depth_dummy * 255).astype(
                        np.uint8)

                    cv2.imwrite(save_path_dummy,
                                cv2.cvtColor(pred_dummy, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(save_path_depth_dummy, pred_depth_dummy)

                if train_dataset is not None and self.opt.add_dummy:
                    train_image = train_dataset.images_dummy.data[
                        self.local_step].detach().cpu().numpy()
                    train_image = (train_image * 255).astype(np.uint8)

                    cv2.imwrite(save_path_train,
                                cv2.cvtColor(train_image, cv2.COLOR_RGB2BGR))

                pbar.set_description(
                    f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                pbar.update(loader.batch_size)

        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        pbar.close()
        if not self.use_loss_as_metric and len(self.metrics) > 0:
            result = self.metrics[0].measure()
            self.stats["results"].append(result if self.best_mode == 'min' else
                                         -result)  # if max mode, use -result
        else:
            self.stats["results"].append(
                average_loss)  # if no metric, choose best by min loss

        for metric in self.metrics:
            self.log(metric.report(), style="blue")
            if self.use_tensorboardX:
                metric.write(self.writer, self.epoch, prefix="evaluate")
            metric.clear()

        if self.opt.add_dummy:
            for metric in self.metrics_dummy:
                self.log(f'Dummy {metric.report()}', style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer,
                                 self.epoch,
                                 prefix="evaluate_dummy")
                metric.clear()

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    def inerf_evaluate_one_epoch(self, loader, name=None, train_dataset=None):
        self.log(f"++> Inerf Evaluate at epoch {self.epoch} ...")

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        total_loss = 0
        for metric in self.metrics:
            metric.clear()

        if self.opt.add_dummy:
            for metric in self.metrics_dummy:
                metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        pbar = tqdm.tqdm(
            total=len(loader) * loader.batch_size,
            bar_format=
            '{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )

        with torch.no_grad():
            self.local_step = 0

            for data in loader:
                self.local_step += 1

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_dummy, preds_depth, pred_depth_dummy, truths, loss, loss_dummy = self.inerf_eval_step(
                        data)

                loss_val = loss.item()
                total_loss += loss_val

                for metric in self.metrics:
                    metric.update(preds, truths)

                for metric in self.metrics_dummy:
                    metric.update(preds_dummy, truths)

                # save image
                save_path = os.path.join(
                    self.workspace, 'validation',
                    f'{name}_{self.local_step:04d}_rgb.png')
                save_path_depth = os.path.join(
                    self.workspace, 'validation',
                    f'{name}_{self.local_step:04d}_depth.png')

                save_path_dummy = os.path.join(
                    self.workspace, 'validation',
                    f'{name}_{self.local_step:04d}_rgb_dummy.png')
                save_path_depth_dummy = os.path.join(
                    self.workspace, 'validation',
                    f'{name}_{self.local_step:04d}_depth_dummy.png')

                if train_dataset is not None:
                    save_path_train = os.path.join(
                        self.workspace, 'validation',
                        f'{name}_{self.local_step:04d}_rgb_train.png')

                #self.log(f"==> Saving validation image to {save_path}")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                if self.opt.color_space == 'linear':
                    preds = linear_to_srgb(preds)

                pred = preds[0].detach().cpu().numpy()
                pred = (pred * 255).astype(np.uint8)

                pred_depth = preds_depth[0].detach().cpu().numpy()
                pred_depth = (pred_depth * 255).astype(np.uint8)

                cv2.imwrite(save_path, cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                cv2.imwrite(save_path_depth, pred_depth)

                pred_dummy = preds_dummy[0].detach().cpu().numpy()
                pred_dummy = (pred_dummy * 255).astype(np.uint8)

                pred_depth_dummy = pred_depth_dummy[0].detach().cpu().numpy()
                pred_depth_dummy = (pred_depth_dummy * 255).astype(np.uint8)

                cv2.imwrite(save_path_dummy,
                            cv2.cvtColor(pred_dummy, cv2.COLOR_RGB2BGR))
                cv2.imwrite(save_path_depth_dummy, pred_depth_dummy)

                pbar.set_description(
                    f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                pbar.update(loader.batch_size)

        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        pbar.close()
        if not self.use_loss_as_metric and len(self.metrics) > 0:
            result = self.metrics[0].measure()
            self.stats["results"].append(result if self.best_mode == 'min' else
                                         -result)  # if max mode, use -result
        else:
            self.stats["results"].append(
                average_loss)  # if no metric, choose best by min loss

        for metric in self.metrics:
            self.log(metric.report(), style="blue")
            if self.use_tensorboardX:
                metric.write(self.writer, self.epoch, prefix="evaluate")
            metric.clear()

        for metric in self.metrics_dummy:
            self.log(f'Dummy {metric.report()}', style="blue")
            if self.use_tensorboardX:
                metric.write(self.writer, self.epoch, prefix="evaluate_dummy")
            metric.clear()

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    def inerf_one_epoch(self, loader, data):
        if self.inerf_global_step < self.opt.inerf_train_steps:
            self.log(
                f"==> Start Training Epoch {self.epoch}, lr={self.inerf_optimizer.param_groups[0]['lr']:.6f} ..."
            )
        else:
            self.log(
                f"==> Start Training Epoch {self.epoch}, lr={self.inerf_optimizer_dummy.param_groups[0]['lr']:.6f} ..."
            )

        total_loss = 0
        if self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        pbar = tqdm.tqdm(
            total=len(loader) * loader.batch_size,
            bar_format=
            '{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )

        self.local_step = 0

        for data in loader:
            self.local_step += 1
            self.inerf_global_step += 1

            if self.inerf_global_step <= self.opt.inerf_train_steps:
                self.inerf_optimizer.zero_grad()
            else:
                self.inerf_optimizer_dummy.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, loss = self.inerf_train_step(data)

            self.scaler.scale(loss).backward()

            if self.inerf_global_step <= self.opt.inerf_train_steps:
                self.scaler.step(self.inerf_optimizer)
            else:
                self.scaler.step(self.inerf_optimizer_dummy)

            self.scaler.update()

            if self.scheduler_update_every_step:
                if self.inerf_global_step <= self.opt.inerf_train_steps:
                    self.inerf_lr_scheduler.step()
                else:
                    self.inerf_lr_scheduler_dummy.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.inerf_global_step > self.opt.inerf_train_steps:
                if self.report_metric_at_train:
                    for metric in self.metrics:
                        metric.update(preds, truths)

                if self.use_tensorboardX:
                    self.writer.add_scalar("inerf_attack_train/loss", loss_val,
                                           self.inerf_global_step)
                    self.writer.add_scalar(
                        "inerf_attack_train/lr",
                        self.inerf_optimizer_dummy.param_groups[0]['lr'],
                        self.inerf_global_step)
            else:
                self.writer.add_scalar("inerf_train/loss", loss_val,
                                       self.inerf_global_step)
                self.writer.add_scalar(
                    "inerf_train/lr",
                    self.inerf_optimizer.param_groups[0]['lr'],
                    self.inerf_global_step)

            pbar.set_description(
                f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
            pbar.update(loader.batch_size)

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        pbar.close()
        if self.report_metric_at_train:
            for metric in self.metrics:
                self.log(metric.report(), style="red")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="train")
                metric.clear()

        if not self.scheduler_update_every_step:
            if self.inerf_global_step <= self.opt.inerf_train_steps:
                if isinstance(self.inerf_lr_scheduler,
                              torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.inerf_lr_scheduler.step(average_loss)
                else:
                    self.inerf_lr_scheduler.step()
            else:
                if isinstance(self.inerf_lr_scheduler_dummy,
                              torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.inerf_lr_scheduler_dummy.step(average_loss)
                else:
                    self.inerf_lr_scheduler_dummy.step()

        self.log(f"==> Finished Epoch {self.epoch}.")

    def save_checkpoint(self,
                        name=None,
                        full=False,
                        best=False,
                        remove_old=True,
                        train_dataset=None):

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
        }

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            if self.opt.add_dummy:
                state['optimizer_dummy'] = self.optimizer_dummy.state_dict()
                state[
                    'lr_scheduler_dummy'] = self.lr_scheduler_dummy.state_dict(
                    )
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()

            if self.opt.add_dummy and not self.opt.inerf:
                state[
                    'train_dataset'] = train_dataset.images_dummy.data if train_dataset is not None else None

        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{self.ckpt_path}/{name}.pth"

            if remove_old:
                self.stats["checkpoints"].append(file_path)

                if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                    old_ckpt = self.stats["checkpoints"].pop(0)
                    if os.path.exists(old_ckpt):
                        os.remove(old_ckpt)

            torch.save(state, file_path)

        else:
            if len(self.stats["results"]) > 0:
                if self.stats["best_result"] is None or self.stats["results"][
                        -1] < self.stats["best_result"]:
                    self.log(
                        f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}"
                    )
                    self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    if self.ema is not None:
                        self.ema.restore()

                    torch.save(state, self.best_path)
            else:
                self.log(
                    f"[WARN] no evaluated results found, skip saving best checkpoint."
                )

    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(
                glob.glob(f'{self.ckpt_path}/{self.name}_ep*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log(
                    "[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(
            checkpoint_dict['model'], strict=False)

        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")

        if self.ema is not None and 'ema' in checkpoint_dict and self.opt.finetune_iter <= 0:
            self.ema.load_state_dict(checkpoint_dict['ema'])

        if model_only:
            return

        self.stats = checkpoint_dict['stats']
        self.stats['checkpoints'] = []

        if self.opt.inerf:
            self.stats["checkpoints"].clear()

        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(
            f"[INFO] load at epoch {self.epoch}, global step {self.global_step}"
        )

        if self.train_dataset and 'train_dataset' in checkpoint_dict:
            if checkpoint_dict[
                    'train_dataset'] is not None and self.opt.finetune_iter <= 0:
                self.train_dataset.images_dummy.data = checkpoint_dict[
                    'train_dataset'].to(self.device)

        if self.optimizer and 'optimizer' in checkpoint_dict and self.opt.finetune_iter <= 0:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")

        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict and self.opt.finetune_iter <= 0:
            try:
                self.lr_scheduler.load_state_dict(
                    checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")
        if self.opt.add_dummy and not self.opt.inerf:
            if self.optimizer_dummy and 'optimizer_dummy' in checkpoint_dict:
                try:
                    self.optimizer_dummy.load_state_dict(
                        checkpoint_dict['optimizer_dummy'])
                    self.log("[INFO] loaded optimizer_dummy.")
                except:
                    self.log("[WARN] Failed to load optimizer_dummy.")

            if self.lr_scheduler_dummy and 'lr_scheduler_dummy' in checkpoint_dict:
                try:
                    self.lr_scheduler_dummy.load_state_dict(
                        checkpoint_dict['lr_scheduler_dummy'])
                    self.log("[INFO] loaded scheduler_dummy.")
                except:
                    self.log("[WARN] Failed to load scheduler_dummy.")

        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")
