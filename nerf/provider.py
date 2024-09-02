import glob
import json
import os

import cv2
import numpy as np
import torch
import tqdm
import trimesh
from scipy.spatial.transform import Rotation, Slerp
from torch.utils.data import DataLoader

from .utils import get_rays


class CameraPoseQuan(torch.nn.Module):

    def __init__(self):
        super(CameraPoseQuan, self).__init__()
        self.q = torch.nn.Parameter(
            torch.normal(0, 1e-6, size=(4, )) + torch.tensor([1., 0., 0., 0.]))
        self.t = torch.nn.Parameter(torch.normal(0, 1e-6, size=(3, )))

    def quaternion_to_matrix(self, quaternions: torch.Tensor) -> torch.Tensor:
        """
        Convert rotations given as quaternions to rotation matrices.

        Args:
            quaternions: quaternions with real part first,
                as tensor of shape (..., 4).

        Returns:
            Rotation matrices as tensor of shape (..., 3, 3).
        """
        r, i, j, k = torch.unbind(quaternions, -1)
        # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
        two_s = 2.0 / (quaternions * quaternions).sum(-1)

        o = torch.stack(
            (
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * r),
                two_s * (i * k + j * r),
                two_s * (i * j + k * r),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * r),
                two_s * (i * k - j * r),
                two_s * (j * k + i * r),
                1 - two_s * (i * i + j * j),
            ),
            -1,
        )
        return o.reshape(quaternions.shape[:-1] + (3, 3))

    # x is start pose
    def forward(self, x):

        exp_i = torch.zeros((4, 4), device=x.device)
        exp_i[:3, :3] = self.quaternion_to_matrix(self.q)
        exp_i[:3, 3] = self.t
        exp_i[3, 3] = 1.

        T_i = torch.matmul(exp_i, x)
        return T_i


# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose, scale=0.33, offset=[0, 0, 0]):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
        [0, 0, 0, 1],
    ],
        dtype=np.float32)
    return new_pose


def visualize_poses(poses, size=0.1):
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)
        o = pos + dir * 3

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b],
                         [b, c], [c, d], [d, a], [pos, o]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show()


def rand_poses(size,
               device,
               radius=1,
               theta_range=[np.pi / 3, 2 * np.pi / 3],
               phi_range=[0, 2 * np.pi]):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, \pi]
        phi_range: [min, max], should be in [0, 2\pi]
    Return:
        poses: [size, 4, 4]
    '''

    def normalize(vectors):
        return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-10)

    thetas = torch.rand(size, device=device) * (
        theta_range[1] - theta_range[0]) + theta_range[0]
    phis = torch.rand(
        size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ],
        dim=-1)  # [B, 3]

    # lookat
    forward_vector = -normalize(centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0).repeat(
        size, 1)  # confused at the coordinate system...
    right_vector = normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float,
                      device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector),
                                   dim=-1)
    poses[:, :3, 3] = centers

    return poses


class NeRFDataset:

    def __init__(self, opt, device, type='train', downscale=1, n_test=150):
        super().__init__()

        self.opt = opt
        self.device = device
        self.type = type  # train, val, test
        self.downscale = downscale
        self.root_path = opt.path
        self.preload = opt.preload  # preload data into GPU
        # camera radius scale to make sure camera are inside the bounding box.
        self.scale = opt.scale
        self.offset = opt.offset  # camera offset
        # bounding box half length, also used as the radius to random sample poses.
        self.bound = opt.bound
        self.fp16 = opt.fp16  # if preload, load into fp16.

        self.training = self.type in ['train', 'all', 'trainval', 'inerf']
        self.num_rays = self.opt.num_rays if self.training else -1

        self.rand_pose = opt.rand_pose

        # auto-detect transforms.json and split mode.
        if os.path.exists(os.path.join(self.root_path, 'transforms.json')):
            # manually split, use view-interpolation for test.
            self.mode = 'colmap'
        elif os.path.exists(
                os.path.join(self.root_path, 'transforms_train.json')):
            self.mode = 'blender'  # provided split
        else:
            raise NotImplementedError(
                f'[NeRFDataset] Cannot find transforms*.json under {self.root_path}'
            )

        # load nerf-compatible format data.
        if self.mode == 'colmap':
            with open(os.path.join(self.root_path, 'transforms.json'),
                      'r') as f:
                transform = json.load(f)
        elif self.mode == 'blender':
            # load all splits (train/valid/test), this is what instant-ngp in fact does...
            if type == 'all':
                transform_paths = glob.glob(
                    os.path.join(self.root_path, '*.json'))
                transform = None
                for transform_path in transform_paths:
                    with open(transform_path, 'r') as f:
                        tmp_transform = json.load(f)
                        if transform is None:
                            transform = tmp_transform
                        else:
                            transform['frames'].extend(tmp_transform['frames'])
            # load train and val split
            elif type == 'trainval':
                with open(
                        os.path.join(self.root_path, f'transforms_train.json'),
                        'r') as f:
                    transform = json.load(f)
                with open(os.path.join(self.root_path, f'transforms_val.json'),
                          'r') as f:
                    transform_val = json.load(f)
                transform['frames'].extend(transform_val['frames'])
            # only load one specified split
            else:
                with open(
                        os.path.join(self.root_path,
                                     f'transforms_{type}.json'), 'r') as f:
                    transform = json.load(f)

        else:
            raise NotImplementedError(f'unknown dataset mode: {self.mode}')

        # load image size
        if 'h' in transform and 'w' in transform:
            self.H = int(transform['h']) // downscale
            self.W = int(transform['w']) // downscale
        else:
            # we have to actually read an image to get H and W later.
            self.H = self.W = None

        # read images
        frames = transform["frames"]
        # frames = sorted(frames, key=lambda d: d['file_path']) # why do I sort...

        # for colmap, manually interpolate a test set.
        if self.mode == 'colmap' and type == 'test':

            # choose two random poses, and interpolate between.
            f0, f1 = np.random.choice(frames, 2, replace=False)
            pose0 = nerf_matrix_to_ngp(np.array(f0['transform_matrix'],
                                                dtype=np.float32),
                                       scale=self.scale,
                                       offset=self.offset)  # [4, 4]
            pose1 = nerf_matrix_to_ngp(np.array(f1['transform_matrix'],
                                                dtype=np.float32),
                                       scale=self.scale,
                                       offset=self.offset)  # [4, 4]
            rots = Rotation.from_matrix(
                np.stack([pose0[:3, :3], pose1[:3, :3]]))
            slerp = Slerp([0, 1], rots)

            self.poses = []
            self.images = None
            for i in range(n_test + 1):
                ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = slerp(ratio).as_matrix()
                pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                self.poses.append(pose)
            self.poses = torch.from_numpy(np.stack(self.poses,
                                                   axis=0))  # [N, 4, 4]
        else:
            # for colmap, manually split a valid set (the first frame).
            if self.mode == 'colmap':
                if type == 'train':
                    frames = frames[1:]
                elif type == 'val' or type == 'inerf':
                    frames_inerf = frames[1:]
                    frames = frames[:1]
                # else 'all' or 'trainval' : use all frames

            self.poses = []
            self.images = []
            self.depths = []
            for f in tqdm.tqdm(frames, desc=f'Loading {type} data'):
                f_path = os.path.join(self.root_path, f['file_path'])
                if self.mode == 'blender' and '.' not in os.path.basename(
                        f_path):
                    f_path += '.png'  # so silly...
                ori_f_path = f_path

                # there are non-exist paths in fox...
                if not os.path.exists(f_path):
                    continue

                pose = np.array(f['transform_matrix'],
                                dtype=np.float32)  # [4, 4]
                pose = nerf_matrix_to_ngp(pose,
                                          scale=self.scale,
                                          offset=self.offset)

                if 'val' == self.type and self.opt.add_dummy:
                    if "hyprsim" in self.opt.workspace:
                        _f_path = f_path.replace('images', 'images_nerf')
                        _f_path = _f_path.replace('.jpg', '.png')
                    elif "scannet" in self.opt.workspace:
                        _f_path = f_path.replace('rgb', 'rgb_nerf')
                        _f_path = _f_path.replace('.jpg', '.png')
                    elif "front3d" in self.opt.workspace:
                        _f_path = f_path.replace('images', 'images_nerf')
                        _f_path = _f_path.replace('.jpg', '.png')
                    if os.path.exists(_f_path):
                        print(f'[INFO] using dummy image: {_f_path}')
                        f_path = _f_path
                    else:
                        raise RuntimeError(
                            f'cannot find dummy image: {_f_path}')

                image = cv2.imread(
                    f_path, cv2.IMREAD_UNCHANGED)  # [H, W, 3] o [H, W, 4]
                if self.H is None or self.W is None:
                    self.H = image.shape[0] // downscale
                    self.W = image.shape[1] // downscale

                # add support for the alpha channel as a mask.
                if image.shape[-1] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

                if image.shape[0] != self.H or image.shape[1] != self.W:
                    image = cv2.resize(image, (self.W, self.H),
                                       interpolation=cv2.INTER_AREA)
                # if self.type == 'inerf':
                #     s = np.sqrt(self.H * self.W /
                #                 self.num_rays)  # only in training, assert num_rays > 0
                #     rH, rW = int(self.H / s), int(self.W / s)
                #     image = cv2.resize(image, (rW, rH))

                image = image.astype(np.float32) / 255  # [H, W, 3/4]

                if 'inerf' == self.type:
                    if "hyprsim" in self.opt.workspace:
                        _f_path = ori_f_path.replace('images', 'depth_inerf')
                        _f_path = _f_path.replace('.jpg', '.png')
                    elif "scannet" in self.opt.workspace:
                        _f_path = ori_f_path.replace('rgb', 'depth_inerf')
                        _f_path = _f_path.replace('.jpg', '.png')
                    elif "front3d" in self.opt.workspace:
                        _f_path = ori_f_path.replace('images', 'depth_inerf')
                        _f_path = _f_path.replace('.jpg', '.png')
                    if os.path.exists(_f_path):
                        print(f'[INFO] using dummy image: {_f_path}')
                        f_path = _f_path
                    else:
                        raise RuntimeError(
                            f'cannot find dummy image: {_f_path}')

                    depth = cv2.imread(f_path, cv2.IMREAD_UNCHANGED)
                    if self.H is None or self.W is None:
                        self.H = depth.shape[0] // downscale
                        self.W = depth.shape[1] // downscale
                    if depth.shape[0] != self.H or depth.shape[1] != self.W:
                        depth = cv2.resize(depth, (self.W, self.H),
                                           interpolation=cv2.INTER_AREA)
                    if self.type == 'inerf':
                        s = np.sqrt(self.H * self.W / self.num_rays
                                    )  # only in training, assert num_rays > 0
                        rH, rW = int(self.H / s), int(self.W / s)
                        depth = cv2.resize(depth, (rW, rH))

                    depth = np.expand_dims(depth, axis=2)
                    depth = depth.astype(np.float32) / 65535  # [H, W, 1]
                    self.depths.append(depth)

                self.poses.append(pose)
                self.images.append(image)

            self.poses = torch.from_numpy(np.stack(self.poses,
                                                   axis=0))  # [N, 4, 4]

            if type == 'val' or type == 'inerf':
                self.poses_inerf = []
                self.images_inerf = []
                for f in tqdm.tqdm(frames_inerf, desc=f'Loading {type} data'):
                    f_path = os.path.join(self.root_path, f['file_path'])
                    if self.mode == 'blender' and '.' not in os.path.basename(
                            f_path):
                        f_path += '.png'  # so silly...

                    # there are non-exist paths in fox...
                    if not os.path.exists(f_path):
                        continue

                    pose = np.array(f['transform_matrix'],
                                    dtype=np.float32)  # [4, 4]
                    pose = nerf_matrix_to_ngp(pose,
                                              scale=self.scale,
                                              offset=self.offset)

                    self.poses_inerf.append(pose)

                    image = cv2.imread(
                        f_path, cv2.IMREAD_UNCHANGED)  # [H, W, 3] o [H, W, 4]
                    if self.H is None or self.W is None:
                        self.H = image.shape[0] // downscale
                        self.W = image.shape[1] // downscale

                    # add support for the alpha channel as a mask.
                    if image.shape[-1] == 3:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    else:
                        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

                    if image.shape[0] != self.H or image.shape[1] != self.W:
                        image = cv2.resize(image, (self.W, self.H),
                                           interpolation=cv2.INTER_AREA)

                    image = image.astype(np.float32) / 255  # [H, W, 3/4]

                    self.images_inerf.append(image)

                self.poses_inerf = torch.from_numpy(
                    np.stack(self.poses_inerf, axis=0))

        # calculate mean radius of all camera poses
        self.radius = self.poses[:, :3, 3].norm(dim=-1).mean(0).item()

        if type == 'inerf' or type == 'val':
            # most similar image in train images
            error_index = np.linalg.norm(
                np.linalg.inv(self.poses_inerf).dot(self.poses[0]) - np.eye(4),
                axis=(1, 2)).argmin()
            print(f'[INFO] inerf pose error index: {error_index}')
            self.inerf_poses = self.poses[0].unsqueeze(0).to(self.device)
            self.camera_poses = [
                CameraPoseQuan().to(self.device)
                for i in range(self.poses.shape[0])
            ]

        if self.images is not None:
            self.images = torch.from_numpy(np.stack(self.images,
                                                    axis=0))  # [N, H, W, C]
            if type == "inerf":
                self.depths = torch.from_numpy(np.stack(
                    self.depths, axis=0))  # [N, H, W, 1]
            if self.opt.add_dummy:
                if 'train' == self.type:
                    self.images_dummy = torch.nn.Parameter(
                        torch.randn_like(self.images).abs().clamp(0, 1))

        if self.preload:
            self.poses = self.poses.to(self.device)
            if self.images is not None:
                # TODO: linear use pow, but pow for half is only available for torch >= 1.10 ?
                if self.fp16 and self.opt.color_space != 'linear':
                    dtype = torch.half
                else:
                    dtype = torch.float
                self.images = self.images.to(dtype).to(self.device)
                if self.opt.add_dummy:
                    if 'train' == self.type:
                        self.images_dummy = torch.nn.Parameter(
                            torch.randn_like(self.images).abs().clamp(0, 1))

        # load intrinsics
        if 'fl_x' in transform or 'fl_y' in transform:
            fl_x = (transform['fl_x']
                    if 'fl_x' in transform else transform['fl_y']) / downscale
            fl_y = (transform['fl_y']
                    if 'fl_y' in transform else transform['fl_x']) / downscale
        elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
            # blender, assert in radians. already downscaled since we use H/W
            fl_x = self.W / (2 * np.tan(transform['camera_angle_x'] / 2)
                             ) if 'camera_angle_x' in transform else None
            fl_y = self.H / (2 * np.tan(transform['camera_angle_y'] / 2)
                             ) if 'camera_angle_y' in transform else None
            if fl_x is None:
                fl_x = fl_y
            if fl_y is None:
                fl_y = fl_x
        else:
            raise RuntimeError(
                'Failed to load focal length, please check the transforms.json!'
            )

        cx = (transform['cx'] / downscale) if 'cx' in transform else (self.W /
                                                                      2)
        cy = (transform['cy'] / downscale) if 'cy' in transform else (self.H /
                                                                      2)

        self.intrinsics = np.array([fl_x, fl_y, cx, cy])

    def collate(self, index):

        B = len(index)  # a list of length 1

        poses = self.poses[index].to(self.device)  # [B, 4, 4]
        if self.type == 'inerf' or self.type == 'val':
            inerf_poses = []
            for i in range(B):
                # forward function, input is the start pose
                inerf_poses.append(self.camera_poses[index[i]](
                    self.inerf_poses[index[i]]))
            new_inerf_poses = torch.stack(inerf_poses, dim=0)

        if self.type == 'inerf':
            s = np.sqrt(self.H * self.W /
                        self.num_rays)  # only in training, assert num_rays > 0
            rH, rW = int(self.H / s), int(self.W / s)

        if self.type == 'inerf':
            rays = get_rays(poses, self.intrinsics / s, rH, rW, -1)
        else:
            rays = get_rays(poses, self.intrinsics, self.H, self.W,
                            self.num_rays)

        if self.type == 'inerf' or self.type == 'val':
            if self.type == 'inerf':
                inerf_rays = get_rays(new_inerf_poses, self.intrinsics / s, rH,
                                      rW, -1, rays)
            else:
                inerf_rays = get_rays(new_inerf_poses, self.intrinsics, self.H,
                                      self.W, self.num_rays)

        results = {
            'H': rH if self.type == 'inerf' else self.H,
            'W': rW if self.type == 'inerf' else self.W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
        }

        if self.type == 'inerf' or self.type == 'val':
            results['inerf_rays_o'] = inerf_rays['rays_o']
            results['inerf_rays_d'] = inerf_rays['rays_d']

        if self.images is not None:
            images = self.images[index].to(self.device)  # [B, H, W, 3/4]
            if self.opt.add_dummy and 'train' == self.type:
                images_dummy = self.images_dummy[index]
            if self.training and 'inerf' != self.type:
                C = images.shape[-1]
                images = torch.gather(images.view(B, -1, C), 1,
                                      torch.stack(C * [rays['inds']],
                                                  -1))  # [B, N, 3/4]
                if 'train' == self.type and self.opt.add_label_noise:
                    images = images + torch.normal(0,
                                                   self.opt.label_noise_std,
                                                   images.shape,
                                                   device=images.device)
                if self.opt.add_dummy and 'train' == self.type:
                    images_dummy = torch.gather(
                        images_dummy.view(B, -1, C), 1,
                        torch.stack(C * [rays['inds']], -1))
            results['images'] = images
            if self.type == "inerf":
                results["depths"] = self.depths[index].to(self.device)
            if self.opt.add_dummy and 'train' == self.type:
                results['images_dummy'] = images_dummy

        return results

    def dataloader(self):
        size = len(self.poses)
        if self.training and self.rand_pose > 0:
            # index >= size means we use random pose.
            size += size // self.rand_pose
        loader = DataLoader(list(range(size)),
                            batch_size=1,
                            collate_fn=self.collate,
                            shuffle=self.training,
                            num_workers=0)
        # an ugly fix... we need to access error_map & poses in trainer.
        loader._data = self
        loader.has_gt = self.images is not None
        return loader
