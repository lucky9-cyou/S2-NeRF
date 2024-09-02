import argparse

import numpy as np
import torch
import torch.optim as optim

from activation import noise_config
from nerf.network import NeRFNetwork
from nerf.provider import NeRFDataset
from nerf.trainer import Trainer
from nerf.utils import LPIPSMeter, PSNRMeter, seed_everything

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('-O',
                        action='store_true',
                        help="equals --fp16 --cuda_ray --preload")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=42)

    ### training options
    parser.add_argument('--iters',
                        type=int,
                        default=60000,
                        help="training iters")
    parser.add_argument('--lr',
                        type=float,
                        default=1e-2,
                        help="initial learning rate")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument(
        '--num_rays',
        type=int,
        default=4096,
        help="num rays sampled per image for each training step")
    parser.add_argument(
        '--max_steps',
        type=int,
        default=1024,
        help="max num steps sampled per ray (only valid when using --cuda_ray)"
    )
    parser.add_argument(
        '--num_steps',
        type=int,
        default=512,
        help="num steps sampled per ray (only valid when NOT using --cuda_ray)"
    )
    parser.add_argument(
        '--max_ray_batch',
        type=int,
        default=4096,
        help=
        "batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)"
    )

    ### network backbone options
    parser.add_argument('--fp16',
                        action='store_true',
                        help="use amp mixed precision training")

    ### dataset options
    parser.add_argument('--color_space',
                        type=str,
                        default='srgb',
                        help="Color space, supports (linear, srgb)")
    parser.add_argument(
        '--preload',
        action='store_true',
        help=
        "preload all data into GPU, accelerate training but use more GPU memory"
    )
    # (the default value is for the fox dataset)
    parser.add_argument(
        '--bound',
        type=float,
        default=2,
        help=
        "assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching."
    )
    parser.add_argument('--scale',
                        type=float,
                        default=0.33,
                        help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--offset',
                        type=float,
                        nargs='*',
                        default=[0, 0, 0],
                        help="offset of camera location")
    parser.add_argument(
        '--dt_gamma',
        type=float,
        default=1 / 128,
        help=
        "dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)"
    )
    parser.add_argument('--min_near',
                        type=float,
                        default=0.2,
                        help="minimum near distance for camera")
    parser.add_argument('--density_thresh',
                        type=float,
                        default=10,
                        help="threshold for density grid to be occupied")
    parser.add_argument(
        '--bg_radius',
        type=float,
        default=-1,
        help="if positive, use a background model at sphere(bg_radius)")
    parser.add_argument(
        '--rand_pose',
        type=int,
        default=-1,
        help=
        "<0 uses no rand pose, =0 only uses rand pose, >0 sample one rand pose every $ known poses"
    )
    parser.add_argument("--eval_interval", type=int, default=20)

    ## noise options
    parser.add_argument('--noise_std',
                        type=float,
                        default=0.0,
                        help="noise std for sigma")
    parser.add_argument('--add_noise',
                        action='store_true',
                        help="add noise to sigma")
    parser.add_argument("--noise_layer",
                        type=int,
                        default=0,
                        help="noise layer")
    parser.add_argument("--add_label_noise",
                        action="store_true",
                        help="add label noise")
    parser.add_argument("--label_noise_std",
                        type=float,
                        default=0.0,
                        help="label noise std")
    parser.add_argument("--noise_type", type=str, default="mix")
    parser.add_argument("--noise_prob", type=float, default=1.0)
    parser.add_argument("--gradient_clip", action='store_true', default=False)
    parser.add_argument("--noise_decay", type=float, default=0.005)
    parser.add_argument("--partial_scale", type=float, default=10000)
    parser.add_argument("--warp_loss",
                        action='store_true',
                        help="add warp loss")
    parser.add_argument("--add_dummy",
                        action='store_true',
                        help="add dummy layer")
    parser.add_argument("--dummy_layer",
                        type=int,
                        default=0,
                        help="dummy layer")
    parser.add_argument("--inerf_train_steps", type=int, default=0)
    parser.add_argument("--inerf_steps", type=int, default=200)
    parser.add_argument("--lambda_label", type=float, default=1.0)
    parser.add_argument("--lambda_grad", type=float, default=1.0)
    parser.add_argument("--inerf",
                        action='store_true',
                        default=False,
                        help="use iNeRF")
    parser.add_argument("--inerf_thr", type=float, default=0.02)
    parser.add_argument("--finetune_iter", type=int, default=0)
    parser.add_argument("--dummy_lr_decay", type=float, default=0.1)
    parser.add_argument("--num_layers_color_dummy", type=int, default=3)

    opt = parser.parse_args()

    opt.iters += opt.finetune_iter

    if opt.O:
        opt.fp16 = True
        opt.preload = True

    print(opt)

    seed_everything(opt.seed)

    noise_config.std = opt.noise_std
    noise_config.prob = opt.noise_prob
    noise_config.total_iter = opt.iters
    noise_config.gradient_clip = opt.gradient_clip
    noise_config.noise_decay = opt.noise_decay
    noise_config.partial_scale = opt.partial_scale

    model = NeRFNetwork(
        opt,
        encoding="hashgrid",
        bound=opt.bound,
        density_scale=1,
        min_near=opt.min_near,
        density_thresh=opt.density_thresh,
        bg_radius=opt.bg_radius,
    )

    print(model)

    criterion = torch.nn.MSELoss(reduction='none')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = NeRFDataset(opt, device=device,
                                type='train') if not opt.inerf else None
    if opt.inerf:
        inerf_train_dataset = NeRFDataset(opt, device=device, type='inerf')
    val_dataset = NeRFDataset(opt, device=device, type='val')

    optimizer = lambda model: torch.optim.Adam(
        model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)

    if opt.add_dummy and not opt.inerf:
        optimizer_dummy = lambda model: torch.optim.Adam(
            model.get_params_dummy(opt.lr) + [{
                'params': train_dataset.images_dummy,
                'lr': opt.lr
            }],
            betas=(0.9, 0.99),
            eps=1e-15)
    else:
        optimizer_dummy = None

    if opt.inerf:
        inerf_optimizer = lambda model: torch.optim.Adam(
            [{
                'params': inerf_train_dataset.camera_poses[i].parameters(),
                'lr': opt.lr
            } for i in range(len(inerf_train_dataset.camera_poses))],
            betas=(0.9, 0.99),
            eps=1e-15)
        inerf_optimizer_dummy = lambda model: torch.optim.Adam(
            model.get_params_dummy(opt.lr), betas=(0.9, 0.99), eps=1e-15)
    else:
        inerf_optimizer = None
        inerf_optimizer_dummy = None

    train_loader = train_dataset.dataloader() if not opt.inerf else None
    valid_loader = val_dataset.dataloader()
    if opt.inerf:
        inerf_train_loader = inerf_train_dataset.dataloader()

    # decay to 0.1 * init_lr at last iter step
    if opt.finetune_iter <= 0:
        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(
            optimizer, lambda iter: 0.1**min(iter / opt.iters, 1))
    else:
        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(
            optimizer, lambda iter: 0.1**min(iter / opt.finetune_iter, 1))

    if opt.add_dummy:
        # scheduler_dummy = lambda optimizer_dummy: optim.lr_scheduler.LambdaLR(
        # optimizer_dummy, lambda iter: 10 / (iter + 1))
        if opt.dummy_lr_decay == 0.1:
            scheduler_dummy = lambda optimizer_dummy: optim.lr_scheduler.LambdaLR(
                optimizer_dummy, lambda iter: 0.1**min(iter / opt.iters, 1))
        elif opt.dummy_lr_decay == 0.001:
            scheduler_dummy = lambda optimizer_dummy: optim.lr_scheduler.LambdaLR(
                optimizer_dummy, lambda iter: 0.001**min(iter / opt.iters, 1))
        else:
            scheduler_dummy = lambda optimizer_dummy: optim.lr_scheduler.LambdaLR(
                optimizer_dummy, lambda iter: 10 / (iter + 1))
        # scheduler_dummy = lambda optimizer_dummy: optim.lr_scheduler.LambdaLR(
        #     optimizer_dummy, lambda iter: 0.995 ** (iter))
    else:
        scheduler_dummy = None

    if opt.inerf:
        inerf_lr_scheduler = lambda inerf_optimizer: optim.lr_scheduler.LambdaLR(
            inerf_optimizer, lambda iter: 0.1**min(iter / opt.inerf_steps, 1))
        inerf_lr_scheduler_dummy = lambda inerf_optimizer_dummy: optim.lr_scheduler.LambdaLR(
            inerf_optimizer_dummy, lambda iter: 0.1**min(
                iter / (opt.inerf_steps - opt.inerf_train_steps), 1))
    else:
        inerf_lr_scheduler = None
        inerf_lr_scheduler_dummy = None

    metrics = [PSNRMeter(), LPIPSMeter(device=device)]
    if opt.add_dummy:
        metrics_dummy = [PSNRMeter(), LPIPSMeter(device=device)]
    else:
        metrics_dummy = None

    trainer = Trainer('ngp',
                      opt,
                      model,
                      device=device,
                      workspace=opt.workspace,
                      optimizer=optimizer,
                      optimizer_dummy=optimizer_dummy,
                      inerf_optimizer=inerf_optimizer,
                      inerf_optimizer_dummy=inerf_optimizer_dummy,
                      criterion=criterion,
                      ema_decay=0.95,
                      fp16=opt.fp16,
                      lr_scheduler=scheduler,
                      lr_scheduler_dummy=scheduler_dummy,
                      inerf_lr_scheduler=inerf_lr_scheduler,
                      inerf_lr_scheduler_dummy=inerf_lr_scheduler_dummy,
                      scheduler_update_every_step=True,
                      metrics=metrics,
                      metrics_dummy=metrics_dummy,
                      use_checkpoint=opt.ckpt,
                      eval_interval=opt.eval_interval,
                      train_dataset=train_dataset)

    max_epoch = np.ceil(opt.iters / len(train_loader)).astype(
        np.int32) if not opt.inerf else trainer.epoch
    trainer.train(train_loader, valid_loader, max_epoch, train_dataset)

    if opt.inerf:
        del train_loader, train_dataset
        trainer.inerf_train(inerf_train_loader, valid_loader, opt.inerf_steps,
                            inerf_train_dataset, val_dataset)

    # also test
    test_loader = NeRFDataset(opt, device=device, type='test').dataloader()

    if test_loader.has_gt:
        trainer.evaluate(test_loader)  # blender has gt, so evaluate it.

    trainer.test(test_loader, write_video=True)  # test and save video

    trainer.save_mesh(resolution=256, threshold=10)
