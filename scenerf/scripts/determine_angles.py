import torch
import numpy as np
from scenerf.models.spherical_mapping import SphericalMapping, pix_2_cam_pts

if __name__ == "__main__":
    """
    Please note that if you want to adjust the code to work with your own datasets, 
    you simply need to modify the "cam_K" and "img_W" and "img_H" parameters accordingly.
    """
    cam_K = torch.tensor([[707.0912,   0.0000, 601.8873],
                          [0.0000, 707.0912, 183.1104],
                          [0.0000,   0.0000,   1.0000]])
    inv_K = torch.inverse(cam_K)
  
    img_W, img_H = 1220, 370
    
    out_img_W = 0 # not important, set to any value
    out_img_H = 0 # not important, set to any value
    mapping = SphericalMapping(
        v_angle_max=0, # not important, set to any value
        v_angle_min=0, # not important, set to any value
        h_angle_max=0, # not important, set to any value
        h_angle_min=0, # not important, set to any value
        img_W=img_W, img_H=img_H, out_img_W=out_img_W, out_img_H=out_img_H)

    #Get pixel coordinates
    meshgrid = np.meshgrid(
                range(img_W), range(img_H), indexing='xy')
    id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
    id_coords = torch.from_numpy(id_coords)
    pix_coords = torch.cat(
        [id_coords[0].reshape(-1, 1), id_coords[1].reshape(-1, 1)], 1).type_as(inv_K)

    # Get any points on the rays through the pixel coordinates
    depth = torch.ones((pix_coords.shape[0])).type_as(inv_K)
    cam_pts = pix_2_cam_pts(pix_coords, inv_K, depth)


    # Get all the possible angles
    v_angle, h_angle, _ = mapping.cam_pts_2_angle(cam_pts)
    print("v_angle_min: ", v_angle.min())
    print("v_angle_max: ", v_angle.max())
    print("h_angle_min: ", h_angle.min())
    print("h_angle_max: ", h_angle.max())
    print("horizontal FOV: ", h_angle.max() - h_angle.min())
    print("vertical FOV: ", v_angle.max() - v_angle.min())
    
    # with 150% increase in FOV: need to add 0.25 FOV to each side
    print("add_fov_ver", 0.25 * (v_angle.max() - v_angle.min()))
    print("add_fov_hor", 0.25 * (h_angle.max() - h_angle.min()))
