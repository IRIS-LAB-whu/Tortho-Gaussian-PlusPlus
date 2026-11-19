#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.gaussian_adam import RowWiseGaussianAdam

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.enable_rowwise_birth_adam = False
        self.preserve_opacity_birth_step = False
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.view_support_enabled = False
        self.obs_tensor = torch.empty(0)
        self.obs_count = torch.empty(0)
        self.support_score = torch.empty(0)
        self.weak_dir = torch.empty(0)
        self.support_score_valid = torch.empty(0, dtype=torch.bool)
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self._capture_view_support_state(),
        )
    
    def restore(self, model_args, training_args):
        view_support_state = None
        if len(model_args) == 14:
            # Legacy checkpoints may carry a final lineage payload.
            (*model_args, view_support_state, _) = model_args
        elif len(model_args) == 13:
            (*model_args, view_support_state) = model_args

        (self.active_sh_degree,
        self._xyz,
        self._features_dc,
        self._features_rest,
        self._scaling,
        self._rotation,
        self._opacity,
        self.max_radii2D,
        xyz_gradient_accum,
        denom,
        opt_dict,
        self.spatial_lr_scale) = model_args
        if any(group.get("visibility_aware_adam", False) for group in opt_dict.get("param_groups", [])):
            raise ValueError("This checkpoint uses a removed experimental optimizer; start a new training run.")
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        if (getattr(training_args, "enable_weak_support", False) or getattr(training_args, "enable_view_support", False)) and view_support_state is not None and view_support_state[0]:
            self._restore_view_support_state(view_support_state)


    def _view_support_device(self):
        if hasattr(self._xyz, "device"):
            return self._xyz.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _view_support_dtype(self):
        if hasattr(self._xyz, "dtype"):
            return self._xyz.dtype
        return torch.float32

    def _capture_view_support_state(self):
        return (
            self.view_support_enabled,
            self.obs_tensor,
            self.obs_count,
            self.support_score,
            self.weak_dir,
            self.support_score_valid,
        )

    def _restore_view_support_state(self, state):
        self.view_support_enabled = bool(state[0])
        self.obs_tensor = state[1].detach().to(self._view_support_device())
        self.obs_count = state[2].detach().to(self._view_support_device())
        self.support_score = state[3].detach().to(self._view_support_device())
        self.weak_dir = state[4].detach().to(self._view_support_device())
        self.support_score_valid = state[5].detach().to(self._view_support_device()).bool()
        self._ensure_view_support_state()

    def enable_view_support_tracking(self):
        self.view_support_enabled = True
        self._ensure_view_support_state()

    def disable_view_support_tracking(self):
        self.view_support_enabled = False
        self.obs_tensor = torch.empty(0)
        self.obs_count = torch.empty(0)
        self.support_score = torch.empty(0)
        self.weak_dir = torch.empty(0)
        self.support_score_valid = torch.empty(0, dtype=torch.bool)

    def _ensure_view_support_state(self):
        if not self.view_support_enabled:
            return
        n_points = self.get_xyz.shape[0]
        device = self._view_support_device()
        dtype = self._view_support_dtype()
        if self.obs_tensor.shape[0] == n_points and self.obs_tensor.device == device:
            return
        self.obs_tensor = torch.zeros((n_points, 3, 3), device=device, dtype=dtype)
        self.obs_count = torch.zeros((n_points), device=device, dtype=dtype)
        self.support_score = torch.ones((n_points), device=device, dtype=dtype)
        self.weak_dir = torch.zeros((n_points, 3), device=device, dtype=dtype)
        if n_points > 0:
            self.weak_dir[:, 0] = 1.0
        self.support_score_valid = torch.zeros((n_points), device=device, dtype=torch.bool)

    def _as_bool_point_mask(self, mask, n_points):
        device = self._view_support_device()
        if mask is None:
            return torch.zeros((n_points), device=device, dtype=torch.bool)
        mask = mask.detach().to(device)
        if mask.dtype == torch.bool:
            mask = mask.reshape(-1)
            bool_mask = torch.zeros((n_points), device=device, dtype=torch.bool)
            copy_count = min(mask.numel(), n_points)
            if copy_count > 0:
                bool_mask[:copy_count] = mask[:copy_count]
            return bool_mask
        bool_mask = torch.zeros((n_points), device=device, dtype=torch.bool)
        if mask.numel() > 0:
            bool_mask[mask.reshape(-1).long()] = True
        return bool_mask

    @torch.no_grad()
    def update_view_support(self, camera_center, visible_mask, ema):
        if not self.view_support_enabled:
            return
        self._ensure_view_support_state()
        n_points = self.get_xyz.shape[0]
        visible_mask = self._as_bool_point_mask(visible_mask, n_points)
        if not visible_mask.any():
            return

        # A = I - vv^T accumulates directions constrained by the current ray.
        # Its null-space is the weakly observed direction for single-sided views.
        xyz = self.get_xyz.detach()[visible_mask]
        center = camera_center.detach().to(device=xyz.device, dtype=xyz.dtype).view(1, 3)
        view_dirs = torch.nn.functional.normalize(xyz - center, dim=1, eps=1e-8)
        eye = torch.eye(3, device=xyz.device, dtype=xyz.dtype).unsqueeze(0)
        obs_matrix = eye - view_dirs.unsqueeze(2) * view_dirs.unsqueeze(1)
        ema = float(ema)
        self.obs_tensor[visible_mask] = ema * self.obs_tensor[visible_mask] + (1.0 - ema) * obs_matrix
        self.obs_count[visible_mask] += 1.0

    @torch.no_grad()
    def update_view_support_scores(self, update_mask=None):
        if not self.view_support_enabled:
            return
        self._ensure_view_support_state()
        n_points = self.get_xyz.shape[0]
        device = self._view_support_device()
        dtype = self._view_support_dtype()

        if update_mask is None:
            valid_mask = self.obs_count > 0
            self.support_score = torch.ones((n_points), device=device, dtype=dtype)
            self.weak_dir = torch.zeros((n_points, 3), device=device, dtype=dtype)
            if n_points > 0:
                self.weak_dir[:, 0] = 1.0
            self.support_score_valid = torch.zeros((n_points), device=device, dtype=torch.bool)
        else:
            # Only visible Gaussians from the current render pass need fresh
            # eigendecomposition; other Gaussians keep their last support state.
            valid_mask = self._as_bool_point_mask(update_mask, n_points)
            valid_mask = torch.logical_and(valid_mask, self.obs_count > 0)
        if not valid_mask.any():
            return

        # F is symmetric positive semi-definite up to numerical noise.
        # support_score = lambda_min / trace(F) is bounded and stable across
        # different EMA magnitudes; single-view support gives a score near 0.
        valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
        for chunk_indices in valid_indices.split(500_000):
            obs_tensor = self.obs_tensor[chunk_indices]
            obs_tensor = 0.5 * (obs_tensor + obs_tensor.transpose(1, 2))
            eigvals, eigvecs = torch.linalg.eigh(obs_tensor)
            lambda_min = eigvals[:, 0].clamp_min(0.0)
            trace = eigvals.sum(dim=1).clamp_min(0.0)
            self.support_score[chunk_indices] = lambda_min / (trace + 1e-8)
            self.weak_dir[chunk_indices] = eigvecs[:, :, 0]
            self.support_score_valid[chunk_indices] = True


    def get_supported_mask(self):
        n_points = self.get_xyz.shape[0]
        if not self.view_support_enabled or self.support_score.shape[0] != n_points:
            return torch.zeros((n_points), device=self._view_support_device(), dtype=torch.bool)
        valid_mask = self.obs_count > 0
        if self.support_score_valid.shape[0] == n_points:
            valid_mask = torch.logical_and(valid_mask, self.support_score_valid)
        return valid_mask

    def view_support_statistics(self):
        n_points = self.get_xyz.shape[0]
        stats = {
            "score_ready": False,
            "mean_support_score": 0.0,
            "min_support_score": 0.0,
            "max_support_score": 0.0,
            "obs_count_mean": 0.0,
            "obs_count_min": 0.0,
            "obs_count_max": 0.0,
            "total_gaussians": n_points,
            "supported_gaussians": 0,
        }
        if not self.view_support_enabled or n_points == 0:
            return stats

        supported_mask = self.get_supported_mask()
        supported_count = int(supported_mask.sum().item())
        stats["supported_gaussians"] = supported_count
        if supported_mask.any():
            score = self.support_score[supported_mask].detach()
            stats["score_ready"] = True
            stats["mean_support_score"] = score.mean().item()
            stats["min_support_score"] = score.min().item()
            stats["max_support_score"] = score.max().item()

        # obs_count is shown over all current Gaussians so never-seen points
        # remain visible in the min statistic.
        obs_count = self.obs_count.detach()
        stats["obs_count_mean"] = obs_count.mean().item()
        stats["obs_count_min"] = obs_count.min().item()
        stats["obs_count_max"] = obs_count.max().item()
        return stats


    def _view_support_extension_from_mask(self, selected_pts_mask, repeat_count=1):
        if not self.view_support_enabled:
            return None
        self._ensure_view_support_state()
        selected_pts_mask = self._as_bool_point_mask(selected_pts_mask, self.obs_tensor.shape[0])
        repeat_count = int(repeat_count)
        return {
            "obs_tensor": self.obs_tensor[selected_pts_mask].repeat(repeat_count, 1, 1),
            "obs_count": self.obs_count[selected_pts_mask].repeat(repeat_count),
            "support_score": self.support_score[selected_pts_mask].repeat(repeat_count),
            "weak_dir": self.weak_dir[selected_pts_mask].repeat(repeat_count, 1),
            "support_score_valid": self.support_score_valid[selected_pts_mask].repeat(repeat_count),
        }

    def _append_view_support(self, extension):
        if not self.view_support_enabled or extension is None:
            return
        self.obs_tensor = torch.cat((self.obs_tensor, extension["obs_tensor"].to(self.obs_tensor.device)), dim=0)
        self.obs_count = torch.cat((self.obs_count, extension["obs_count"].to(self.obs_count.device)), dim=0)
        self.support_score = torch.cat((self.support_score, extension["support_score"].to(self.support_score.device)), dim=0)
        self.weak_dir = torch.cat((self.weak_dir, extension["weak_dir"].to(self.weak_dir.device)), dim=0)
        self.support_score_valid = torch.cat((self.support_score_valid, extension["support_score_valid"].to(self.support_score_valid.device).bool()), dim=0)

    def _prune_view_support(self, valid_points_mask):
        if not self.view_support_enabled or self.obs_tensor.shape[0] != valid_points_mask.shape[0]:
            return
        self.obs_tensor = self.obs_tensor[valid_points_mask]
        self.obs_count = self.obs_count[valid_points_mask]
        self.support_score = self.support_score[valid_points_mask]
        self.weak_dir = self.weak_dir[valid_points_mask]
        self.support_score_valid = self.support_score_valid[valid_points_mask]

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.enable_rowwise_birth_adam = getattr(training_args, "enable_rowwise_birth_adam", False)
        self.preserve_opacity_birth_step = getattr(training_args, "preserve_opacity_birth_step", False)
        if self.enable_rowwise_birth_adam and self.optimizer_type != "default":
            raise ValueError("--enable_rowwise_birth_adam only supports --optimizer_type default")
        if self.enable_rowwise_birth_adam and getattr(training_args, "inherit_parent_adam_state", False):
            raise ValueError("--enable_rowwise_birth_adam and --inherit_parent_adam_state are mutually exclusive")
        if self.preserve_opacity_birth_step and not self.enable_rowwise_birth_adam:
            raise ValueError("--preserve_opacity_birth_step requires --enable_rowwise_birth_adam")
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        if getattr(training_args, "enable_weak_support", False) or getattr(training_args, "enable_view_support", False):
            self.enable_view_support_tracking()
        else:
            self.disable_view_support_tracking()

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self.optimizer_type == "default":
            if self.enable_rowwise_birth_adam:
                self.optimizer = RowWiseGaussianAdam(
                    l, lr=0.0, eps=1e-15, max_steps=training_args.iterations
                )
            else:
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    @staticmethod
    def weak_support_score_to_rgb(scores):
        """Map the theoretical score range [0, 1/3] to fixed RGB colors."""
        scores = np.asarray(scores, dtype=np.float32)
        normalized = np.clip(scores / (1.0 / 3.0), 0.0, 1.0).reshape(-1)
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)

        # Fixed anchors make diagnostic colors comparable across datasets.
        anchors = np.asarray(
            [
                [0.7059, 0.0157, 0.1490],
                [0.9569, 0.4275, 0.2627],
                [0.9961, 0.8784, 0.5451],
                [0.4000, 0.7608, 0.6471],
                [0.1961, 0.3137, 0.7059],
            ],
            dtype=np.float32,
        )
        position = normalized * (anchors.shape[0] - 1)
        lower = np.minimum(position.astype(np.int64), anchors.shape[0] - 2)
        fraction = (position - lower).reshape(-1, 1)
        rgb = anchors[lower] * (1.0 - fraction) + anchors[lower + 1] * fraction
        return rgb.reshape(scores.shape + (3,))

    @classmethod
    def _save_weak_support_score_chart(cls, path, scores, valid_mask):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "Generating the weak-support score chart requires opencv-python"
            ) from exc

        mkdir_p(os.path.dirname(path))
        score_min = 0.0
        score_max = 1.0 / 3.0
        valid_mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        finite_valid_mask = np.logical_and(valid_mask, np.isfinite(scores))
        valid_scores = scores[finite_valid_mask]

        width, height = 1280, 820
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX

        def put_text(text, x, y, scale=0.58, color=(30, 30, 30), thickness=1):
            cv2.putText(
                canvas,
                text,
                (int(x), int(y)),
                font,
                scale,
                color,
                thickness,
                cv2.LINE_AA,
            )

        put_text("Weak-support score distribution", 80, 42, scale=0.9, thickness=2)
        put_text(
            "Fixed cross-dataset color range: 0.000000 (weak) to 0.333333 (strong)",
            80,
            76,
            scale=0.58,
        )

        plot_left, plot_right = 100, 1190
        plot_top, plot_bottom = 115, 500
        plot_width = plot_right - plot_left
        plot_height = plot_bottom - plot_top
        bin_count = 64
        histogram, bin_edges = np.histogram(
            np.clip(valid_scores, score_min, score_max),
            bins=bin_count,
            range=(score_min, score_max),
        )
        ratios = histogram.astype(np.float64) / max(valid_scores.size, 1)
        max_ratio = float(ratios.max()) if ratios.size > 0 else 0.0

        for tick_index in range(5):
            fraction = tick_index / 4.0
            y = int(plot_bottom - fraction * plot_height)
            cv2.line(canvas, (plot_left, y), (plot_right, y), (225, 225, 225), 1)
            label_value = fraction * max_ratio
            put_text("{:.1%}".format(label_value), 25, y + 5, scale=0.45)

        if valid_scores.size > 0 and max_ratio > 0.0:
            centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            bar_colors = cls.weak_support_score_to_rgb(centers)
            for index, ratio in enumerate(ratios):
                left = int(plot_left + index * plot_width / bin_count)
                right = max(left + 1, int(plot_left + (index + 1) * plot_width / bin_count) - 1)
                top = int(plot_bottom - (ratio / max_ratio) * plot_height)
                rgb = np.rint(bar_colors[index] * 255.0).astype(np.uint8)
                bgr = tuple(int(channel) for channel in rgb[::-1])
                cv2.rectangle(canvas, (left, top), (right, plot_bottom), bgr, -1)
        else:
            put_text("No observed Gaussians with a valid score", 390, 310, scale=0.7)

        cv2.rectangle(canvas, (plot_left, plot_top), (plot_right, plot_bottom), (40, 40, 40), 1)
        put_text("Fraction of valid Gaussians per bin", 100, 105, scale=0.48)

        score_ticks = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, score_max]
        for score_tick in score_ticks:
            x = int(plot_left + (score_tick - score_min) / (score_max - score_min) * plot_width)
            cv2.line(canvas, (x, plot_bottom), (x, plot_bottom + 7), (40, 40, 40), 1)
            label = "1/3" if np.isclose(score_tick, score_max) else "{:.2f}".format(score_tick)
            put_text(label, x - 16, plot_bottom + 28, scale=0.43)

        colorbar_top, colorbar_bottom = 575, 625
        gradient_scores = np.linspace(score_min, score_max, plot_width + 1, dtype=np.float32)
        gradient_rgb = np.rint(cls.weak_support_score_to_rgb(gradient_scores) * 255.0).astype(np.uint8)
        gradient_bgr = gradient_rgb[:, ::-1]
        canvas[colorbar_top:colorbar_bottom, plot_left:plot_right + 1] = np.repeat(
            gradient_bgr[np.newaxis, :, :],
            colorbar_bottom - colorbar_top,
            axis=0,
        )
        cv2.rectangle(
            canvas,
            (plot_left, colorbar_top),
            (plot_right, colorbar_bottom),
            (40, 40, 40),
            1,
        )
        put_text("Score-to-color reference", plot_left, colorbar_top - 14, scale=0.55)
        for score_tick in score_ticks:
            x = int(plot_left + (score_tick - score_min) / (score_max - score_min) * plot_width)
            cv2.line(canvas, (x, colorbar_bottom), (x, colorbar_bottom + 7), (40, 40, 40), 1)
            label = "0.333" if np.isclose(score_tick, score_max) else "{:.2f}".format(score_tick)
            put_text(label, x - 18, colorbar_bottom + 27, scale=0.43)

        invalid_rgb = np.asarray([0.5, 0.5, 0.5], dtype=np.float32)
        invalid_bgr = tuple(int(channel) for channel in np.rint(invalid_rgb[::-1] * 255.0))
        cv2.rectangle(canvas, (100, 675), (145, 700), invalid_bgr, -1)
        cv2.rectangle(canvas, (100, 675), (145, 700), (40, 40, 40), 1)
        invalid_count = int(valid_mask.size - finite_valid_mask.sum())
        put_text(
            "gray = unobserved / invalid: {} ({:.2%})".format(
                invalid_count,
                invalid_count / max(valid_mask.size, 1),
            ),
            160,
            696,
            scale=0.55,
        )

        if valid_scores.size > 0:
            quantiles = np.percentile(valid_scores, [5, 25, 50, 75, 95])
            put_text(
                "valid: {} / {}    mean: {:.6f}    min/max: {:.6f} / {:.6f}".format(
                    valid_scores.size,
                    valid_mask.size,
                    float(valid_scores.mean()),
                    float(valid_scores.min()),
                    float(valid_scores.max()),
                ),
                100,
                744,
                scale=0.56,
            )
            put_text(
                "percentiles p05/p25/p50/p75/p95: {:.6f} / {:.6f} / {:.6f} / {:.6f} / {:.6f}".format(
                    *[float(value) for value in quantiles]
                ),
                100,
                780,
                scale=0.53,
            )
        else:
            put_text("valid: 0 / {}".format(valid_mask.size), 100, 744, scale=0.56)

        if not cv2.imwrite(path, canvas):
            raise RuntimeError("Failed to write weak-support score chart: {}".format(path))

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)


    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                if self.enable_rowwise_birth_adam and name == "opacity" and not self.preserve_opacity_birth_step:
                    current_step = int(stored_state["step"].item())
                    stored_state["birth_step"] = torch.full(
                        (tensor.shape[0],),
                        current_step,
                        dtype=torch.int32,
                        device=tensor.device,
                    )

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                if self.enable_rowwise_birth_adam:
                    stored_state["birth_step"] = stored_state["birth_step"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]
        self._prune_view_support(valid_points_mask)


    def _optimizer_state_extension_from_mask(self, selected_pts_mask, repeat_count=1):
        """Copy selected parents' Adam moments in the same order as their children."""
        n_points = self.get_xyz.shape[0]
        selected_pts_mask = self._as_bool_point_mask(selected_pts_mask, n_points)
        state_extensions = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is None or "exp_avg" not in stored_state or "exp_avg_sq" not in stored_state:
                continue
            if stored_state["exp_avg"].shape[0] != n_points or stored_state["exp_avg_sq"].shape[0] != n_points:
                continue
            repeat_dims = (int(repeat_count),) + (1,) * (stored_state["exp_avg"].dim() - 1)
            state_extensions[group["name"]] = {
                "exp_avg": stored_state["exp_avg"][selected_pts_mask].repeat(*repeat_dims),
                "exp_avg_sq": stored_state["exp_avg_sq"][selected_pts_mask].repeat(*repeat_dims),
            }
        return state_extensions

    def cat_tensors_to_optimizer(self, tensors_dict, state_extensions=None):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                group_state_extension = state_extensions.get(group["name"]) if state_extensions is not None else None
                if group_state_extension is None:
                    exp_avg_extension = torch.zeros_like(extension_tensor)
                    exp_avg_sq_extension = torch.zeros_like(extension_tensor)
                else:
                    exp_avg_extension = group_state_extension["exp_avg"]
                    exp_avg_sq_extension = group_state_extension["exp_avg_sq"]
                    if exp_avg_extension.shape != extension_tensor.shape or exp_avg_sq_extension.shape != extension_tensor.shape:
                        raise ValueError("Adam state extension shape must match the densification tensor shape")

                # Keep the existing parameter-tensor step; only new rows' moments differ.
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], exp_avg_extension), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], exp_avg_sq_extension), dim=0)
                if self.enable_rowwise_birth_adam:
                    current_step = int(stored_state["step"].item())
                    birth_step_extension = torch.full(
                        (extension_tensor.shape[0],),
                        current_step,
                        dtype=torch.int32,
                        device=extension_tensor.device,
                    )
                    stored_state["birth_step"] = torch.cat(
                        (stored_state["birth_step"], birth_step_extension), dim=0
                    )

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii, new_view_support=None, new_optimizer_state=None):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d, new_optimizer_state)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._append_view_support(new_view_support)

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2, inherit_parent_adam_state=False):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)
        new_view_support = self._view_support_extension_from_mask(selected_pts_mask, repeat_count=N)
        new_optimizer_state = (
            self._optimizer_state_extension_from_mask(selected_pts_mask, repeat_count=N)
            if inherit_parent_adam_state else None
        )
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii, new_view_support, new_optimizer_state)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, inherit_parent_adam_state=False):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]
        new_view_support = self._view_support_extension_from_mask(selected_pts_mask)
        new_optimizer_state = (
            self._optimizer_state_extension_from_mask(selected_pts_mask)
            if inherit_parent_adam_state else None
        )
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii, new_view_support, new_optimizer_state)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii, inherit_parent_adam_state=False):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent, inherit_parent_adam_state=inherit_parent_adam_state)
        self.densify_and_split(grads, max_grad, extent, inherit_parent_adam_state=inherit_parent_adam_state)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
