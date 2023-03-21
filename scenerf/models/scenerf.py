import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR

from scenerf.loss.depth_metrics import compute_depth_errors
from scenerf.loss.ss_loss import compute_l1_loss
from scenerf.models.pe import PositionalEncoding
from scenerf.models.ray_som_kl import RaySOM
from scenerf.models.resnetfc import ResnetFC
from scenerf.models.unet2d_sphere import UNet2DSphere
from scenerf.models.utils import (
    compute_direction_from_pixels, sample_rays_viewdir, sample_pix_features,
    cam_pts_2_cam_pts, pix_2_cam_pts,
    cam_pts_2_pix, sample_feats_2d,
    sample_rays_gaussian)
from scenerf.models.spherical_mapping import SphericalMapping


class scenerf(pl.LightningModule):
    def __init__(
            self,
            dataset,
            som_sigma,
            lr=1e-5,
            weight_decay=0,
            img_size=(1220, 370),
            n_rays=1200,
            max_infer_depth=120,
            max_sample_depth=100,
            eval_depth=80,
            std=2.5,
            n_gaussians=4,
            n_pts_uni=32,
            n_pts_per_gaussian=8,
            sampling_method="uniform",
            batch_size=1,
            add_fov_hor=0, add_fov_ver=0,
            sphere_H=452, sphere_W=1500,
            use_color=True,
            use_reprojection=True,
    ):
        super().__init__()
        
        self.use_color = use_color
        self.use_reprojection = use_reprojection
        self.dataset = dataset
        self.lr = lr
        self.weight_decay = weight_decay
        self.img_size = img_size
        self.sampling_method = sampling_method
        self.density_head = "softplus"
        self.min_depth = 0.1
        self.depth_window = 100

        self.n_rays = n_rays

        self.n_pts_uni = n_pts_uni
        self.n_gaussians = n_gaussians
        self.n_pts_per_gaussian = n_pts_per_gaussian
        self.std = std
        self.batch_size = batch_size


        self.ray_batch_size = 100
        self.max_infer_depth = max_infer_depth
        self.max_sample_depth = max_sample_depth
        self.eval_depth = eval_depth
        self.sample_distance = 0.2
        self.epoch_start_depth_guide = 1

        self.scene_size = np.array([51.2, 51.2, 6.4])
        self.vox_origin = np.array([0, -25.6, -2])
    
  
        feature = 256
        self.out_img_W = sphere_W
        self.out_img_H = sphere_H
        self.spherical_mapping = SphericalMapping(
            v_angle_max=104.7294 + add_fov_ver,
            v_angle_min=75.4815 - add_fov_ver,
            h_angle_max=131.1128 + add_fov_hor,
            h_angle_min=49.5950 - add_fov_hor,
            img_W=img_size[0], img_H=img_size[1], out_img_W=self.out_img_W, out_img_H=self.out_img_H)

         
        self.net_rgb = UNet2DSphere.build(
            out_feature=feature, out_img_W=self.out_img_W, out_img_H=self.out_img_H)

        self.save_hyperparameters()

        self.pe = PositionalEncoding(
            num_freqs=6,
            include_input=True)

        self.mlp = ResnetFC(
            d_in=39 + 3,
            d_out=4,
            n_blocks=3,
            d_hidden=512,
            d_latent=2480
        )

        self.mlp_gaussian = ResnetFC(
            d_in=39 + 3,
            d_out=2,
            n_blocks=3,
            d_hidden=512,
            d_latent=2480
        )
        self.ray_som = RaySOM(som_sigma=som_sigma)
       
        

    def forward(self, batch, step_type):
        """
        pts_3d: bs, n_pts, 3
        """
        img_input = batch["img_inputs"]

        bs = img_input.shape[0]

        T_velo2cams = batch["T_velo_2_cam"]
        T_cam2velo = torch.inverse(T_velo2cams[0])
        cam_K = batch['cam_K'][0]
        inv_K = torch.inverse(cam_K)

        pix_coords, out_pix_coords, _ = self.spherical_mapping.from_pixels(inv_K=inv_K)
        x_rgbs = self.net_rgb(img_input, pix=pix_coords,
                              pix_sphere=out_pix_coords)

        total_loss_reprojection = 0
        total_loss_color = 0
        total_loss_kl = 0
        total_min_som_vars = 0

        total_min_stds = 0
        total_loss_dist2closest_gauss = 0
        
        for i in range(bs):
            
            T_source2targets = batch['T_source2targets'][i]
            T_source2infers = batch['T_source2infers'][i]
            img_sources = batch['img_sources'][i]
            img_targets = batch['img_targets'][i]
            loc2d_with_depths = batch['loc2d_with_depths'][i]
            lidar_depths = batch['lidar_depths'][i]
            n_sources = len(img_sources)
            
            x_rgb = {}
            for k in x_rgbs:
                x_rgb[k] = x_rgbs[k][i]
             
            cam_K = batch['cam_K'][i]
            inv_K = torch.inverse(cam_K)

            
            for sid in range(n_sources):

                img_target = img_targets[sid]
                img_source = img_sources[sid]
            
                T_source2infer = T_source2infers[sid]
                T_source2target = T_source2targets[sid]
                loc2d_with_depth = loc2d_with_depths[sid]
                lidar_depth = lidar_depths[sid]

                ret = self.process_single_source(
                    self.n_rays,
                    x_rgb=x_rgb,
                    # x_sphere=x_sphere,
                    cam_K=cam_K, inv_K=inv_K,
                    img_source=img_source, img_target=img_target,
                    T_source2target=T_source2target, T_source2infer=T_source2infer,
                    T_cam2velo=T_cam2velo,
                    step_type=step_type)

                total_min_som_vars += ret['min_som_vars'].mean()
                total_loss_kl += ret['loss_kl'].mean()

                total_loss_dist2closest_gauss += ret['loss_dist2closest_gauss'].mean()
                total_min_stds += ret['min_stds'].mean()
                total_loss_reprojection += ret['loss_reprojection'].mean()
                total_loss_color += ret['loss_color'].mean()

                # ====== Depth evaluation ======
                gt_sampled_pixels_infer = loc2d_with_depth.float()
                gt_depth_infer = lidar_depth
                render_out_dict = self.render_rays_batch(
                    cam_K,
                    T_source2infer,
                    x_rgb,
                    ray_batch_size=gt_sampled_pixels_infer.shape[0],
                    sampled_pixels=gt_sampled_pixels_infer)
                pred_depth_infer = render_out_dict['depth']
                self.evaluate_depth(
                    step_type, gt_depth_infer, pred_depth_infer)

        # ==== Combine all the losses
        total_loss = 0


        total_loss_reprojection /= bs
        if self.use_reprojection:
            total_loss += total_loss_reprojection
            self.log(step_type + "/loss_reprojection",
                 total_loss_reprojection.detach(), on_epoch=True, sync_dist=True)


        total_loss_color /= bs
        if self.use_color:
            total_loss += total_loss_color
            self.log(step_type + "/loss_color",
                    total_loss_color.detach(), on_epoch=True, sync_dist=True)

        # SOM loss
        total_loss_kl /= bs
        total_loss += total_loss_kl
        self.log(step_type + "/loss_som_kl",
                 total_loss_kl.detach(), on_epoch=True, sync_dist=True)

        total_min_som_vars /= bs
        self.log(step_type + "/min_som_vars",
                 total_min_som_vars.detach(), on_epoch=True, sync_dist=True)

        # dist 2 closest gaussian loss
        total_loss_dist2closest_gauss /= bs
        total_loss += total_loss_dist2closest_gauss * 0.01  #
        self.log(step_type + "/loss_dist2closest_gauss", total_loss_dist2closest_gauss.detach(), on_epoch=True,
                 sync_dist=True)


        self.log(step_type + "/total_loss", total_loss.detach(),
                 on_epoch=True, sync_dist=True)
        return {
            "total_loss": total_loss
        }

    def process_single_source(self,
                              n_grids,
                              x_rgb,
                              cam_K, inv_K,
                              img_source, img_target,
                              T_source2target, T_source2infer,
                              T_cam2velo,
                              step_type):


        xs = torch.arange(start=0, end=self.img_size[0], step=2).type_as(cam_K)
        ys = torch.arange(start=0, end=self.img_size[1], step=2).type_as(cam_K)
        grid_x, grid_y = torch.meshgrid(xs, ys)

        sampled_pixels = torch.cat([
            grid_x.unsqueeze(-1),
            grid_y.unsqueeze(-1)
        ], dim=2).reshape(-1, 2)

        perm = torch.randperm(sampled_pixels.shape[0])
        idx = perm[:n_grids]
        pix_source = sampled_pixels[idx, :]

        render_out_dict = self.render_rays_batch(
            cam_K,
            T_source2infer,
            x_rgb,
            T_cam2velo=T_cam2velo,
            ray_batch_size=pix_source.shape[0],
            sampled_pixels=pix_source)
        
        depth_source_rendered = render_out_dict['depth']
        color_rendered = render_out_dict['color']
        loss_kl = render_out_dict['loss_kl']
        gaussian_means = render_out_dict['gaussian_means']
        gaussian_stds = render_out_dict['gaussian_stds']
        som_vars = render_out_dict['som_vars']

        weights_at_depth = render_out_dict["weights_at_depth"]


        closest_pts_to_depths = render_out_dict['closest_pts_to_depths']
        self.log(step_type + "depth/closest_pts_to_depth", closest_pts_to_depths.mean().detach(),
                 on_epoch=True, sync_dist=True)
        self.log(step_type + "depth/weights_at_depth", weights_at_depth.mean().detach(), on_epoch=True,
                 sync_dist=True)

        diff = torch.abs(gaussian_means -
                         depth_source_rendered.unsqueeze(-1).detach())
        min_diff, gaussian_idx = torch.min(diff, dim=1)
        loss_dist2closest_gauss = min_diff
        min_stds = torch.gather(gaussian_stds, 1, gaussian_idx.unsqueeze(-1))
        min_som_vars = torch.gather(som_vars, 1, gaussian_idx.unsqueeze(-1))

        self.log(step_type + "_som/dist_2_closest_gaussian", min_diff.mean().detach(), on_epoch=True,
                 sync_dist=True)
        self.log(step_type + "_som/closest_std",
                 min_stds.mean().detach(), on_epoch=True, sync_dist=True)

        sampled_color_source = sample_pix_features(pix_source, img_source)
        loss_color = torch.abs(color_rendered - sampled_color_source.T)

        loss_reprojection = self.compute_reprojection_loss(
            pix_source, sampled_color_source, depth_source_rendered,
            img_target, inv_K, cam_K, T_source2target)
        loss_reprojection = loss_reprojection  # * reliable_rays

        ret = {
            "loss_kl": loss_kl,
            "loss_dist2closest_gauss": loss_dist2closest_gauss,
            "loss_reprojection": loss_reprojection,
            "loss_color": loss_color,
            "weights_at_depth": weights_at_depth, 
            "min_som_vars": min_som_vars,
            "min_stds": min_stds
        }

        return ret

    def evaluate_depth(self, step_type, gt_depth, pred_depth, log=True, ret_mean=True):
        depth_errors = []

        depth_error = compute_depth_errors(
            gt_depth.reshape(-1).detach().cpu().numpy(),
            pred_depth.reshape(-1).detach().cpu().numpy(),
        )

        depth_errors.append(depth_error)

        if ret_mean:
            agg_depth_errors = np.array(depth_errors).mean(0)
        else:
            agg_depth_errors = np.array(depth_errors).sum(0)
        metric_list = ["abs_rel", "sq_rel",
                       "rmse", "rmse_log", "a1", "a2", "a3"]
        
        if not log:
            return agg_depth_errors

        for i_metric, metric in enumerate(metric_list):
            key = step_type + "depth/{}".format(metric)

            self.log(key, agg_depth_errors[i_metric],
                    on_epoch=True, sync_dist=True)

 
    def compute_reprojection_loss(
            self,
            pix_source, sampled_color_source,
            depth_rendered,
            img_target,
            inv_K, cam_K, T_source2target):
        loss_reprojections = []
        cam_source_pts = pix_2_cam_pts(pix_source, inv_K, depth_rendered)
        cam_pts_target = cam_pts_2_cam_pts(cam_source_pts, T_source2target)

        pix_target = cam_pts_2_pix(cam_pts_target, cam_K)
        mask = cam_pts_target[:, 2] > 0

        pix_source = pix_source[mask, :]
        pix_target = pix_target[mask, :]
        sampled_color_source = sampled_color_source[:, mask]


        # ===== Sample the colors at 2d locations =====
        sampled_color_target = sample_pix_features(pix_target, img_target)

        sampled_color_target_identity_reprojection = sample_pix_features(
            pix_source, img_target)

        loss_reprojection = compute_l1_loss(
            sampled_color_source, sampled_color_target)
        loss_identity_reprojection = compute_l1_loss(
            sampled_color_source, sampled_color_target_identity_reprojection)
        loss_identity_reprojection += torch.randn(
            loss_identity_reprojection.shape, device=self.device) * 0.00001

        loss_reprojections.append(loss_reprojection)
        loss_reprojections.append(loss_identity_reprojection)

        loss_reprojections = torch.stack(loss_reprojections)
        loss_reprojections = torch.min(loss_reprojections, dim=0)[0]
     
        return loss_reprojections

    def step(self, batch, step_type):
        out_dict = self.forward(batch, step_type)
        return out_dict['total_loss']

    def render_rays_batch(self, cam_K,
                          T_source2infer,
                          x_rgb,
                        #   x_sphere,
                          depth_window=100,
                          T_cam2velo=None,
                          sampled_pixels=None,
                          ray_batch_size=128):

        inv_K = torch.inverse(cam_K)

        depth_rendereds = []
        gaussian_means = []
        gaussian_stds = []
        weights_at_depth = []
        closest_pts_to_depths = []
        som_vars = []
        densities = []
        weights = []
        alphas = []
        depth_volumes = []
        color_rendereds = []


        cnt = 0
        loss_kl = []

        for start_i in range(0, sampled_pixels.shape[0], ray_batch_size):
            end_i = start_i + ray_batch_size
            batch_sampled_pixels = sampled_pixels[start_i:end_i]

            ret = self.batchify_depth_and_color(
                T_source2infer, x_rgb,
                # x_sphere, 
                batch_sampled_pixels, cam_K, inv_K,
                depth_window=depth_window, T_cam2velo=T_cam2velo)

            color_rendereds.append(ret['color'])
            depth_rendereds.append(ret['depth'])
            gaussian_means.append(ret['gaussian_means'])
            gaussian_stds.append(ret['gaussian_stds'])
            weights_at_depth.append(ret['weights_at_depth'])
            closest_pts_to_depths.append(ret['closest_pts_to_depth'])
            loss_kl.append(ret['loss_kl'])
            densities.append(ret['density'])
            weights.append(ret['weights'])
            alphas.append(ret['alphas'])
            depth_volumes.append(ret['depth_volume'])
            som_vars.append(ret['som_vars'])

            cnt += 1

        depth_rendereds = torch.cat(depth_rendereds, dim=0)
        gaussian_means = torch.cat(gaussian_means, dim=0)
        gaussian_stds = torch.cat(gaussian_stds, dim=0)
        weights_at_depth = torch.cat(weights_at_depth, dim=0)
        closest_pts_to_depths = torch.cat(closest_pts_to_depths, dim=0)
        loss_kl = torch.cat(loss_kl, dim=0)
        densities = torch.cat(densities, dim=0)
        weights = torch.cat(weights, dim=0)
        alphas = torch.cat(alphas, dim=0)
        depth_volumes = torch.cat(depth_volumes, dim=0)
        som_vars = torch.cat(som_vars, dim=0)
        color_rendereds = torch.cat(color_rendereds, dim=0)
        ret = {
            "depth": depth_rendereds,
            "color": color_rendereds,
            "gaussian_means": gaussian_means,
            "gaussian_stds": gaussian_stds,
            "weights_at_depth": weights_at_depth,
            "closest_pts_to_depths": closest_pts_to_depths,
            "loss_kl": loss_kl,
            "alphas": alphas,
            "som_vars": som_vars,
            "densities": densities,
            "weights": weights,
            "depth_volumes": depth_volumes
        }

        return ret

    def density_activation(self, density_logit):
        if self.density_head == "relu":
            density_chunk = F.relu(density_logit)
        elif self.density_head == "softplus":
            softplus = nn.Softplus(beta=1)
            density_chunk = softplus(density_logit - 1)
        else:
            density_chunk = density_logit
        return density_chunk

    def batchify_density(self,
                         x_rgb,
                         cam_pts,
                         cam_K,
                         mlp,
                         T_cam2velo,
                         pts_chunk=192 * 100):
        densities = []
        # colors = []

        for ci in range(0, cam_pts.shape[0], pts_chunk):
            cam_pts_chunk = cam_pts[ci:ci + pts_chunk]

            density_chunk = self.predict(
                mlp=mlp, cam_pts=cam_pts_chunk, x_rgb=x_rgb, cam_K=cam_K, T_cam2velo=T_cam2velo)

            densities.append(density_chunk)

        return {
            'density': torch.cat(densities, dim=0),
        }

    def predict(self, mlp, 
        cam_pts, x_rgb, 
        # x_sphere, 
        cam_K, T_cam2velo, viewdir, output_type="density"):
        saved_shape = cam_pts.shape
        # print(cam_pts.shape)
        cam_pts = cam_pts.reshape(-1, 3)
        projected_pix = cam_pts_2_pix(cam_pts, cam_K)


        _, pix_sphere_coords, _ = self.spherical_mapping.from_pixels(
            inv_K=torch.inverse(cam_K),
            pix_coords=projected_pix)
       
        pe = self.pe(cam_pts)
        

        feats_2d_sphere = [sample_feats_2d(x_rgb["1_1"].unsqueeze(0), pix_sphere_coords, (self.out_img_W, self.out_img_H))]
        for scale in [2, 4, 8, 16]:
            key = "1_{}".format(scale)
            feats_2d_sphere.append(sample_feats_2d(x_rgb[key].unsqueeze(0), pix_sphere_coords, (self.out_img_W//scale, self.out_img_H//scale)))
        
        feats_2d_sphere = torch.cat(feats_2d_sphere, dim=-1)
    
        viewdir = viewdir.unsqueeze(1).expand(-1, saved_shape[1], -1).reshape(-1, 3)

        x_in = torch.cat([feats_2d_sphere, pe, viewdir], dim=-1)

        if output_type == "density":      
            mlp_output = mlp(x_in)
            color = torch.sigmoid(mlp_output[..., :3])
            density = self.density_activation(mlp_output[..., 3:4])
          
            if len(saved_shape) == 3:
                density = density.reshape(saved_shape[0], saved_shape[1])
                color = color.reshape(saved_shape[0], saved_shape[1], 3)
            return density, color
        elif output_type == "offset":
            mlp_output = mlp(x_in)
            residual = mlp_output
            if len(saved_shape) == 3:
                residual = residual.reshape(saved_shape[0], saved_shape[1], 2)
            return residual

    def predict_gaussian_means_and_stds(self, T_source2infer, unit_direction, n_gaussians, 
        x_rgb, 
        # x_sphere,
        cam_K, base_std, T_cam2velo, viewdir):
        n_rays = unit_direction.shape[0]
        step = self.max_sample_depth * 1.0 / self.n_gaussians

        gaussian_means_sensor_distance = torch.linspace(
            step / 2,
            self.max_sample_depth - step / 2,
            steps=n_gaussians
        ).type_as(cam_K)

        gaussian_means_sensor_distance = gaussian_means_sensor_distance.reshape(
            1, n_gaussians, 1).expand(n_rays, -1, 1)

        direction = unit_direction.reshape(
            n_rays, 1, 3).expand(-1, n_gaussians, -1)
        gaussian_means_pts = gaussian_means_sensor_distance * direction

        gaussian_means_pts_infer = cam_pts_2_cam_pts(
            gaussian_means_pts.reshape(-1, 3), T_source2infer)
        gaussian_means_pts_infer = gaussian_means_pts_infer.reshape(
            n_rays, n_gaussians, 3)


        output = self.predict(
            mlp=self.mlp_gaussian,
            cam_pts=gaussian_means_pts_infer,
            x_rgb=x_rgb,
            # x_sphere=x_sphere,
            cam_K=cam_K,
            viewdir=viewdir,
            T_cam2velo=T_cam2velo,
            output_type="offset")

        gaussian_means_offset = output[:, :, 0]
        gaussian_stds_offset = output[:, :, 1]

        gaussian_means_sensor_distance = gaussian_means_sensor_distance.squeeze(
            -1) + gaussian_means_offset

        gaussian_means_sensor_distance = torch.relu(
            gaussian_means_sensor_distance) + 1.5  # avoid negative distance
        gaussian_stds_sensor_distance = torch.relu(
            gaussian_stds_offset + base_std) + 1.5
        
        return gaussian_means_sensor_distance, gaussian_stds_sensor_distance

    def batchify_depth_and_color(
            self, T_source2infer, x_rgb, 
            # x_sphere, 
            batch_sampled_pixels,
            cam_K, inv_K, depth_window, T_cam2velo):
        depths = []
        ret = {}
        n_rays = batch_sampled_pixels.shape[0]
        unit_direction = compute_direction_from_pixels(
            batch_sampled_pixels, inv_K)
        
        cam_pts_uni, depth_volume_uni, sensor_distance_uni, viewdir = sample_rays_viewdir(
            inv_K, T_source2infer,
            self.img_size,
            sampling_method="uniform",
            sampled_pixels=batch_sampled_pixels,
            n_pts_per_ray=self.n_pts_uni,
            max_sample_depth=self.max_sample_depth)

        gaussian_means_sensor_distance, gaussian_stds_sensor_distance = self.predict_gaussian_means_and_stds(
            T_source2infer,
            unit_direction, self.n_gaussians,
            x_rgb=x_rgb,
            cam_K=cam_K,
            T_cam2velo=T_cam2velo,
            base_std=self.std, 
            viewdir=viewdir)

        
        cam_pts_gauss, depth_volume_gauss, sensor_distance_gauss = sample_rays_gaussian(
            T_cam2cam=T_source2infer,
            n_rays=n_rays,
            unit_direction=unit_direction,
            gaussian_means_sensor_distance=gaussian_means_sensor_distance,
            gaussian_stds_sensor_distance=gaussian_stds_sensor_distance,
            n_gaussians=self.n_gaussians, n_pts_per_gaussian=self.n_pts_per_gaussian,
            max_sample_depth=self.max_sample_depth)

        if self.n_pts_uni > 0:  
            cam_pts = torch.cat([cam_pts_uni, cam_pts_gauss],
                                dim=1)  # n_rays, n_pts 3
            depth_volume = torch.cat(
                [depth_volume_uni, depth_volume_gauss], dim=1)  # n_rays, n_pts
            sensor_distance = torch.cat(
                [sensor_distance_uni, sensor_distance_gauss], dim=1)  # n_rays, n_pts
        elif self.n_pts_per_gaussian == 1:
            cam_pts = cam_pts_uni
            depth_volume = depth_volume_uni
            sensor_distance = sensor_distance_uni
        else:
            cam_pts = cam_pts_gauss
            depth_volume = depth_volume_gauss
            sensor_distance = sensor_distance_gauss

        sorted_indices = torch.argsort(sensor_distance, dim=1)

        sensor_distance = torch.gather(
            sensor_distance, dim=1, index=sorted_indices)  # n_rays, n_pts
        depth_volume = torch.gather(
            depth_volume, dim=1, index=sorted_indices)  # n_rays, n_pts
        cam_pts = torch.gather(
            cam_pts, dim=1, index=sorted_indices.unsqueeze(-1).expand(-1, -1, 3))

        density, colors = self.predict(mlp=self.mlp, 
                                       cam_pts=cam_pts.detach(),
                                       viewdir=viewdir,
                                       x_rgb=x_rgb, 
                                       cam_K=cam_K, T_cam2velo=T_cam2velo)

        rendered_out = self.render_depth_and_color(
            density, sensor_distance, depth_volume,
            colors=colors)

        depths = rendered_out['depth_rendered']
        colors = rendered_out['color']

        alphas = rendered_out['alphas']
        weights_at_depth = rendered_out['weights_at_depth']
        closest_pts_to_depth = rendered_out['closest_pts_to_depth']
        weights = rendered_out['weights']

        loss_kl, som_means, som_vars = self.ray_som(
            gaussian_means_sensor_distance,
            gaussian_stds_sensor_distance,
            sensor_distance,
            alphas,
        )

        ret['depth'] = depths
        ret['color'] = colors
        ret['loss_kl'] = loss_kl
        ret['weights_at_depth'] = weights_at_depth
        ret['som_vars'] = som_vars
        ret['gaussian_means'] = gaussian_means_sensor_distance
        ret['gaussian_stds'] = gaussian_stds_sensor_distance
        ret['depth_window'] = depth_window
        ret['closest_pts_to_depth'] = closest_pts_to_depth
        ret['alphas'] = alphas
        ret['density'] = density
        ret['depth_volume'] = depth_volume
        ret['weights'] = weights

        return ret



    def render_depth_and_color(self,
                     density, sensor_distance, depth_volume, colors):

        sensor_distance[sensor_distance < 0] = 0
        deltas = torch.zeros_like(sensor_distance)
        deltas[:, 0] = sensor_distance[:, 0]
        deltas[:, 1:] = sensor_distance[:, 1:] - sensor_distance[:, :-1]
        alphas = 1 - torch.exp(-deltas * density)

    
        ret = {
            "alphas": alphas
        }

        alphas_shifted = torch.cat(
            [torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1
        )  # (B, K+1) = [1, a1, a2, ...]
        T_alphas = torch.cumprod(alphas_shifted, -1)  # (B)

        weights = alphas * T_alphas[:, :-1]  # (B, K)

    
        depth_rendered = torch.sum(weights * depth_volume, -1)
        color_rendered = torch.sum(weights.unsqueeze(-1) * colors, -2) # (B, 3) 
    
 
        diff = depth_rendered.unsqueeze(-1) - depth_volume
        abs_diff = torch.abs(diff)
        closest_pts_to_depth, weights_at_depth_idx = torch.min(abs_diff, dim=1)

        weights_at_depth = torch.gather(
            weights, dim=1, index=weights_at_depth_idx.unsqueeze(-1)).squeeze()
 



        ret['color'] = color_rendered
        ret['weights_at_depth'] = weights_at_depth
        ret['closest_pts_to_depth'] = closest_pts_to_depth
        ret['weights'] = weights
        ret['alphas'] = alphas
        ret['density'] = density
        ret['depth_volume'] = depth_volume
        ret['depth_rendered'] = depth_rendered
        return ret

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = ExponentialLR(optimizer, gamma=0.95)
        return [optimizer], [scheduler]