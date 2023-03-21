import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F


def pix_2_cam_pts(pix, inv_K, depth):
    """
    pix: (B, 2)
    inv_K: (3, 3)
    depth: (B,)
    """
    homo_pix = torch.cat([pix, torch.ones_like(pix)[:, :1]], dim=1)
    cam_pts = (inv_K @ homo_pix.T).T
    cam_pts = depth.view(-1, 1) * cam_pts

    return cam_pts


def cam_pts_2_cam_pts(cam_ptx_from, T):
    """
    cam_ptx_from: B, 3
    """
    ones = torch.ones(cam_ptx_from.shape[0], 1, device=cam_ptx_from.device)
    homo_cam_ptx_from = torch.cat([cam_ptx_from, ones], dim=1).float()
    homo_cam_pts_to = (T @ homo_cam_ptx_from.T).T
    cam_pts_to = homo_cam_pts_to[:, :3]

    return cam_pts_to


def cam_pts_2_pix(cam_pts, K):
    """
    cam_pts: (B, 3)
    K: (3, 3)
    ------
    return
    pix: (B, 2)
    """
    homo_pix = (K @ cam_pts.T).T
    pix = torch.ones(cam_pts.shape[0], 2, device=cam_pts.device) * -1.0
    pix = homo_pix[:, :2] / (homo_pix[:, 2:] + 1e-5)
    return pix


class SphericalMapping(nn.Module):

    def __init__(self,
                 img_W, img_H, out_img_W, out_img_H,
                 # NOTE: These numbers are calculated by running: python scenerf/scripts/determine_angles.py
                 v_angle_max=104.7294,
                 v_angle_min=75.4815,
                 h_angle_max=131.1128,
                 h_angle_min=49.5950,
                 ):

        super(SphericalMapping, self).__init__()
        self.img_W = img_W
        self.img_H = img_H
        self.out_img_W = out_img_W
        self.out_img_H = out_img_H


        self.v_angle_max = v_angle_max
        self.v_angle_min = v_angle_min
        self.h_angle_max = h_angle_max
        self.h_angle_min = h_angle_min
        self.h_fov = abs(self.h_angle_max - self.h_angle_min)
        self.v_fov = abs(self.v_angle_max - self.v_angle_min)



    def from_cam_pts(self, cam_pts, T_cam2velo):
        velo_pts = cam_pts_2_cam_pts(cam_pts, T_cam2velo).squeeze(-1)

        pix_sphere_coords, distance = self.from_velo_pts(velo_pts=velo_pts)
        return None, pix_sphere_coords, distance

    def from_pixels(self, inv_K, pix_coords=None):
        if pix_coords is None:
            meshgrid = np.meshgrid(
                range(self.img_W), range(self.img_H), indexing='xy')
            id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
            id_coords = torch.from_numpy(id_coords)
            pix_coords = torch.cat(
                [id_coords[0].reshape(-1, 1), id_coords[1].reshape(-1, 1)], 1).type_as(inv_K)
        
        depth = torch.ones((pix_coords.shape[0])).type_as(inv_K)
        cam_pts = pix_2_cam_pts(pix_coords, inv_K, depth) 
        pix_sphere_coords, distance = self.cam_pts_2_sphere_coords(cam_pts)
        
        return pix_coords, pix_sphere_coords, distance
    
    def cam_pts_2_angle(self, cam_pts):
        x = cam_pts[:, 0]
        y = cam_pts[:, 1]
        z = cam_pts[:, 2]
        distance = torch.linalg.norm(cam_pts, ord=2, dim=1)
        v_angle = torch.acos(-y/distance)/ math.pi * 180 # wrt minus y direction
        h_angle = 180 - torch.atan2(z, x)/ math.pi * 180 # wrt to x direction
        return v_angle, h_angle, distance

    def cam_pts_2_sphere_coords(self, cam_pts):

        v_angle, h_angle, distance = self.cam_pts_2_angle(cam_pts)
    
        proj_x =  (h_angle - self.h_angle_min) / self.h_fov
        proj_y = (v_angle - self.v_angle_min) / self.v_fov
        out_pix_coords = torch.zeros((cam_pts.shape[0], 2)).type_as(cam_pts)
        out_pix_coords[:, 0] = proj_x * (self.out_img_W - 1)
        out_pix_coords[:, 1] = proj_y * (self.out_img_H - 1)

        
        return torch.round(out_pix_coords).long(), distance



def get_sphere_feature(x, pix, pix_sphere, scale, out_img_W, out_img_H):
    out_W, out_H = round(out_img_W/scale), round(out_img_H/scale)
    map_sphere = torch.zeros((out_W, out_H, 2)).type_as(x) - 1.0
    pix_sphere_scale = pix_sphere // scale
    
    pix_scale = pix * 1.0 / scale
    map_sphere[pix_sphere_scale[:, 0], pix_sphere_scale[:, 1], :] = pix_scale
    map_sphere = map_sphere.reshape(-1, 2)
    
    map_sphere[:, 0] /= x.shape[3] 
    map_sphere[:, 1] /= x.shape[2] 
    map_sphere = map_sphere * 2 - 1
    map_sphere = map_sphere.reshape(1, 1, -1, 2)
    
    feats = F.grid_sample(
        x,
        map_sphere,
        align_corners=False,
        mode='bilinear'
    )  
    feats = feats.reshape(feats.shape[0], feats.shape[1], out_W, out_H)
    feats = feats.permute(0, 1, 3, 2)

    return feats
