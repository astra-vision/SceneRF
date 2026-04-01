from scenerf.data.bundlefusion.bundlefusion_dm import BundlefusionDM
from scenerf.data.utils import fusion
from scenerf.models.utils import sample_rel_poses_bf

import torch
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
import click
import pickle
from scenerf.loss.depth_metrics import compute_depth_errors


torch.set_grad_enabled(False)

def tsdf2occ(tsdf, th=0.25, max_th=0.2, voxel_size=0.04):
    occ = np.zeros(tsdf.shape)
    th_indivi = (voxel_size/2 + np.arange(96).reshape(1, 1, 96) * voxel_size) * th
    th_indivi[th_indivi < voxel_size] = voxel_size
    th_indivi[th_indivi > max_th] = max_th
    occ[(np.abs(tsdf) < th_indivi) & (np.abs(tsdf) != 255)] = 1
    return occ

def read_rgb(path):
    img = Image.open(path).convert("RGB")

    img = np.array(img, dtype=np.float32, copy=False) / 255.0

    return img

def evaluate_depth(gt_depth, pred_depth):
    depth_errors = []

    
    depth_error = compute_depth_errors(
        gt=gt_depth.reshape(-1).detach().cpu().numpy(),
        pred=pred_depth.reshape(-1).detach().cpu().numpy(),
    )

    depth_errors.append(depth_error)

    
    agg_depth_errors = np.array(depth_errors).sum(0)

    
    return agg_depth_errors



@click.command()
@click.option('--n_gpus', default=1, help='number of GPUs')
@click.option('--bs', default=1, help='Batch size')
@click.option('--n_workers_per_gpu', default=3, help='number of workers per GPU')
@click.option('--dataset', default='bf', help='bf or tum_rgbd dataset to eval on')
@click.option('--root', default="/gpfsdswork/dataset/bundlefusion", help='path to dataset folder')
@click.option('--recon_save_dir', default="")
@click.option('--angle', default=30)
@click.option('--step', default=0.2)
@click.option('--max_distance', default=2.1, help='max pose sample distance')
def main(root, dataset, bs, n_gpus, n_workers_per_gpu, recon_save_dir, max_distance, step, angle):

    data_module = BundlefusionDM(
        dataset,
        root=root,
        batch_size=int(bs / n_gpus),
        num_workers=int(n_workers_per_gpu),
        n_sources=1,
    )
    data_module.setup()
    data_loader = data_module.val_dataloader(shuffle=False)

    cnt = 0
    
 
    for batch in tqdm(data_loader):
        cnt += 1
        

        for i in range(bs):
            cam_K = batch['cam_K_depth'][i].numpy()
            
            frame_id = batch['frame_id'][i]
           
            sequence = batch['sequence'][i]
            
            
            tsdf_save_dir = os.path.join(recon_save_dir, "tsdf", sequence)
            os.makedirs(tsdf_save_dir, exist_ok=True)
            save_filepath = os.path.join(tsdf_save_dir, "{}.pkl".format(frame_id))
 


            voxel_size = 0.04
            sx, sy, sz = 4.8, 4.8, 3.84
            scene_size = (sx, sy, sz)
            vox_origin = (-sx / 2, -sy / 2, 0)
            vol_bnds = np.zeros((3,2))
            vol_bnds[:,0] = vox_origin
            vol_bnds[:,1] = vox_origin + np.array([scene_size[0], scene_size[1], scene_size[2]])

            tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=voxel_size, trunc_margin=10)


            rel_poses = sample_rel_poses_bf(angle, max_distance, step)
            for (step, angle), rel_pose in tqdm(rel_poses.items()):
                
                T_source2infer = rel_pose

                depth_save_dir = os.path.join(recon_save_dir, "depth", sequence)
                render_rgb_save_dir = os.path.join(recon_save_dir, "render_rgb", sequence)

                depth_filepath = os.path.join(depth_save_dir,"{}_{:.2f}_{:.2f}.npy".format(frame_id, step, angle))
                render_rgb_filepath = os.path.join(render_rgb_save_dir, "{}_{:.2f}_{:.2f}.png".format(frame_id, step, angle))

                depth = np.load(depth_filepath)
                rgb = read_rgb(render_rgb_filepath) * 255
                
                tsdf_vol.integrate(rgb, depth, cam_K, T_source2infer, obs_weight=1.)
           
            verts, faces, norms, colors = tsdf_vol.get_mesh()
            tsdf_grid, _ = tsdf_vol.get_volume() 

            data = {
                    "tsdf_grid": tsdf_grid,
                    "verts": verts,
                    "faces": faces,
                    "norms": norms,
                    "colors": colors,
                }
            
            with open(save_filepath, "wb") as handle:
                pickle.dump(data, handle)
                print("wrote to", save_filepath)
            
            

if __name__ == "__main__":
    main()
