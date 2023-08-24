import math
import torch
import torch.nn.functional as F


def sample_rel_poses_bf(angle=0, max_distance=2.1, step=0.2):
    steps = torch.arange(start=0, end=max_distance, step=step)
    
    rel_poses = {}
    for step in steps:
        angles = [0]
        if angle != 0:
            angles += [-angle, angle]
        print(step, angles)
        for angle in angles:
            rel_pose = torch.eye(4)
            rel_pose[2, 3] += step
            rad = angle/180 * math.pi
            rot_matrix_y = torch.eye(4)
            rot_matrix_y[:3, :3] = torch.tensor([
                [math.cos(rad), 0, math.sin(rad)],
                [0, 1, 0],
                [-math.sin(rad), 0, math.cos(rad)]
            ])    
            rel_poses[(step, angle)] = rot_matrix_y @ rel_pose
    return rel_poses


def sample_rel_poses(step=0.5, angle=0, max_distance=10.1):
    angles = [0]
    if angle != 0:
        angles += [angle, -angle]
    steps = torch.arange(start=0, end=max_distance, step=step)
    rel_poses = {}
    
    for step in steps:
        for angle in angles:
            rad = angle/180 * math.pi
            rel_pose = torch.eye(4)
            rel_pose[2, 3] += step
          
            rot_matrix_y = torch.eye(4)
            rot_matrix_y[:3, :3] = torch.tensor([
                [math.cos(rad), 0, math.sin(rad)],
                [0, 1, 0],
                [-math.sin(rad), 0, math.cos(rad)]
            ])    
            rel_poses[(step, angle)] = rot_matrix_y @ rel_pose
    return rel_poses


def weighted_uniform_sampling(d_min, d_max, unit_direction, weights):
    n_rays, n_fine, _ = unit_direction.shape
    device = unit_direction.device
    n_coarse = weights.shape[1]

    weights = weights.detach() + 1e-5  # Prevent division by zero
    pdf = weights / torch.sum(weights, -1, keepdim=True)  # (n_rays, n_coarse)
    cdf = torch.cumsum(pdf, -1)  # (B, n_coarse)
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)  # (n_rays, n_coarse+1)

    u = torch.rand(n_rays, n_fine, dtype=torch.float32, device=device)  # (n_rays, n_fine)
    inds = torch.searchsorted(cdf, u, right=True).float() - 1.0  # (n_rays, n_fine)
    inds = torch.clamp_min(inds, 0.0)

    # step = (d_max - d_min) / n_pts_per_ray
    distance_steps = (inds + torch.rand_like(inds)) / n_coarse  # (n_rays, n_fine, 1)
    sensor_distance_sampled = d_min + (d_max - d_min) * distance_steps.unsqueeze(-1)

    cam_pts = sensor_distance_sampled * unit_direction

    return cam_pts, sensor_distance_sampled.squeeze(-1)


def uniform_sampling(d_min, d_max, unit_direction):
    n_rays, n_pts_per_ray, _ = unit_direction.shape
    step = (d_max - d_min) / n_pts_per_ray
    sensor_distance_sampled = torch.linspace(d_min, d_max,
                                             steps=n_pts_per_ray,
                                             device=unit_direction.device) \
        .reshape(1, n_pts_per_ray, 1) \
        .expand(n_rays, -1, -1)

    noise = torch.rand_like(sensor_distance_sampled) * step
    sensor_distance_sampled = sensor_distance_sampled + noise
    

    cam_pts = sensor_distance_sampled * unit_direction

    return cam_pts, sensor_distance_sampled.squeeze(-1)



def log_sampling(d_min, d_max, unit_direction):
    n_rays, n_pts_per_ray, _ = unit_direction.shape
    step = (d_max - d_min) / n_pts_per_ray
    d_i = d_min + torch.arange(n_pts_per_ray - 1, -1, -1, device=unit_direction.device) * (
                d_max - d_min) / n_pts_per_ray

    d_i = d_i.reshape(1, n_pts_per_ray, 1).expand(n_rays, -1, -1)

    noise = torch.rand_like(d_i) * step
    d_i = d_i + noise

    sensor_distance_sampled = d_max - torch.log(d_i - d_min + 1) / math.log(d_max - d_min + 1) * (d_max - d_min)
    
    cam_pts = sensor_distance_sampled * unit_direction

    return cam_pts, sensor_distance_sampled.squeeze()


def sample_rays_viewdir(
        inv_K, T_cam2cam,
        img_size,
        sampling_method="uniform",  # uniform, log
        sampled_pixels=None,
        max_sample_depth=80,
        n_pts_per_ray=256,
        weights=None):
    """
    pix: (n_rays, 2)
    T: (4, 4)
    """
    device = inv_K.device
    if sampled_pixels is None:
        sampled_pixels = torch.rand(n_rays, 2, device=device)
        sampled_pixels[:, 0] = sampled_pixels[:, 0] * img_size[0]
        sampled_pixels[:, 1] = sampled_pixels[:, 1] * img_size[1]
        sampled_pixels = sampled_pixels.reshape(-1, 2)

    n_rays = sampled_pixels.shape[0]

    # Unproject pixels into cam coords to get the direction
    homo_pix = torch.cat([sampled_pixels, torch.ones_like(sampled_pixels)[:, :1]], dim=1)
    viewdir = (inv_K[:3, :3] @ homo_pix.T).T
    cam_pts_direction = viewdir.reshape(n_rays, 1, 3).expand(-1, n_pts_per_ray,
                                                                       -1)  # n_rays, n_pts_per_ray, 3
    unit_direction = F.normalize(cam_pts_direction, dim=2)  # n_rays, n_pts_per_ray, 3

    if sampling_method == "uniform":
        if weights is not None:
            cam_pts, sensor_distance_sampled = weighted_uniform_sampling(
                d_min=0.2,
                d_max=max_sample_depth,
                weights=weights,
                unit_direction=unit_direction)
        else:
            cam_pts, sensor_distance_sampled = uniform_sampling(
                d_min=0.2,
                d_max=max_sample_depth,
                unit_direction=unit_direction)
    elif sampling_method == "log":
        cam_pts, sensor_distance_sampled = log_sampling(d_min=0.2, d_max=max_sample_depth,
                                                        unit_direction=unit_direction)
    else:
        raise "Undefined sampling method"

    # cam_pts = sensor_distance_source * unit_direction
    depth = cam_pts[:, :, 2]

    ones = torch.ones(n_rays, n_pts_per_ray, 1).type_as(cam_pts)
    homo_cam_pts = torch.cat([cam_pts, ones], dim=2).float()

    # Change to camera coord of the other frame    
    homo_pts_infer = (T_cam2cam @ homo_cam_pts.reshape(-1, 4).T).T
    homo_pts_infer = homo_pts_infer.reshape(n_rays, n_pts_per_ray, 4)
    pts_cam = homo_pts_infer[:, :, :3]
    
    
    viewdir_infer = (T_cam2cam[:3, :3] @ viewdir.T).T
    
    # print(depth.shape, sensor_distance_source.shape)
    return pts_cam, depth, sensor_distance_sampled, viewdir_infer



def compute_direction_from_pixels(sampled_pixels, inv_K):
    # Unproject pixels into cam coords to get the direction\
    homo_pix = torch.cat([sampled_pixels, torch.ones_like(sampled_pixels)[:, :1]], dim=1)
    directions = (inv_K[:3, :3] @ homo_pix.T).T
    unit_direction = F.normalize(directions, dim=1)  # n_rays, 3
    return unit_direction



def sample_rays_gaussian(
        T_cam2cam,
        n_rays,
        unit_direction,
        gaussian_means_sensor_distance,
        gaussian_stds_sensor_distance,
        max_sample_depth=60,
        n_gaussians=4,
        n_pts_per_gaussian=8):
    """
    pix: (n_rays, 2)
    T: (4, 4)
    # """
   
    n_pts_per_ray = n_gaussians * n_pts_per_gaussian

    cam_pts_direction = unit_direction.reshape(n_rays, 1, 3).expand(-1, n_pts_per_ray, -1)  # n_rays, n_pts_per_ray, 3

    sensor_distance_sampled = gaussian_means_sensor_distance.repeat_interleave(n_pts_per_gaussian, dim=1)
    std = gaussian_stds_sensor_distance.repeat_interleave(n_pts_per_gaussian, dim=1)


    noise = torch.normal(
        mean=torch.zeros(sensor_distance_sampled.shape),
        std=torch.ones(sensor_distance_sampled.shape)
    ).type_as(sensor_distance_sampled)

    sensor_distance_sampled = sensor_distance_sampled + noise * std
    sensor_distance_sampled[sensor_distance_sampled < 0.1] = 0.1


    cam_pts = sensor_distance_sampled.unsqueeze(-1) * cam_pts_direction

    depth_volume = cam_pts[:, :, 2]

    ones = torch.ones(n_rays, n_pts_per_ray, 1).type_as(cam_pts)
    homo_cam_pts = torch.cat([cam_pts, ones], dim=2).float()

    # Change to camera coord of the other frame    
    homo_pts_infer = (T_cam2cam @ homo_cam_pts.reshape(-1, 4).T).T
    homo_pts_infer = homo_pts_infer.reshape(n_rays, n_pts_per_ray, 4)
    pts_cam = homo_pts_infer[:, :, :3]

    return pts_cam, depth_volume, sensor_distance_sampled


def sample_feats_2d(x_rgb, projected_pix, img_size=(1220, 370)):
    """
    x_rgb: (d, 370, 1220)
    projected_pix: (N, 2)
    """
    projected_pix = (projected_pix / torch.tensor(img_size).type_as(projected_pix).reshape(1, 2)) * 2 - 1
    projected_pix = projected_pix.reshape(1, 1, -1, 2)
    feats_2d = F.grid_sample(
        x_rgb,
        projected_pix,
        align_corners=False,
        mode='bilinear',
        padding_mode="zeros"
    )  # [1, d, n_rays, n_pts_per_rays]
    feats_2d = feats_2d.reshape(feats_2d.shape[1], -1).T
    return feats_2d


def sample_pix_features(pix, img):
    """
    pix: B, 2 # the 2 columns store x, y coords
    img: C, H, W
    -------------
    return 
    color_bilinear: 3, B
    """
    pix = pix.float()
    pix_t = torch.ones_like(pix)  # B, 2
    pix_t[:, 0] = (pix[:, 0] / (img.shape[2] - 1) - 0.5) * 2
    pix_t[:, 1] = (pix[:, 1] / (img.shape[1] - 1) - 0.5) * 2

    color_bilinear = F.grid_sample(
        img.unsqueeze(0),
        pix_t.unsqueeze(0).unsqueeze(2).float(),
        align_corners=False,
        mode='bilinear', padding_mode='zeros').squeeze()

    return color_bilinear


def cam_pts_2_cam_pts(cam_ptx_from, T):
    """
    cam_ptx_from: B, 3
    """
    ones = torch.ones(cam_ptx_from.shape[0], 1, device=cam_ptx_from.device)
    homo_cam_ptx_from = torch.cat([cam_ptx_from, ones], dim=1).float()
    # print(T_cam_minus1_0.dtype, homo_cam_minus1_pts.dtype)
    homo_cam_pts_to = (T @ homo_cam_ptx_from.T).T
    cam_pts_to = homo_cam_pts_to[:, :3]

    return cam_pts_to


def pix_2_cam_pts(pix, inv_K, depth):
    """
    pix: (B, 2)
    inv_K: (3, 3)
    depth: (B,)
    """
    homo_pix = torch.cat([pix, torch.ones_like(pix)[:, :1]], dim=1)
    cam_pts = (inv_K @ homo_pix.T).T
    cam_pts = depth.reshape(-1, 1) * cam_pts

    return cam_pts


def cam_pts_2_pix(cam_pts, K):
    """
    cam_pts: (B, 3)
    K: (3, 3)
    ------
    return
    pix: (B, 2)
    """
    # print(K.dtype, cam_pts.dtype)
    homo_pix = (K @ cam_pts.T).T
    mask = homo_pix[:, 2] > 0
    # homo_pix[mask, 2] = 1.0

    pix = torch.ones(cam_pts.shape[0], 2, device=cam_pts.device) * -1.0
    # pix = homo_pix[:, :2] / (homo_pix[:, 2:] + 1e-5)
    pix[mask, :] = homo_pix[mask, :2] / (homo_pix[mask, 2:])
    # pix = homo_pix[mask, :2] / (homo_pix[mask, 2:])
    return pix



def depth2disp(depth, min_depth=0.1, max_depth=100):
    """Convert depth to disp
    """
    depth = torch.clamp(depth, min=min_depth, max=max_depth)
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = 1 / depth
    disp = scaled_disp - min_disp / (max_disp - min_disp)

    return disp
