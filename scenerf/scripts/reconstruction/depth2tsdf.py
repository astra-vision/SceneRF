from scenerf.data.semantic_kitti.kitti_dm import KittiDataModule
from tqdm import tqdm
import os
import click
import torch
import imageio
from PIL import Image
import numpy as np
from scenerf.models.utils import  sample_rel_poses
from scenerf.data.utils import fusion


def read_depth(depth_filename):
    depth = imageio.imread(depth_filename) / 256.0  # numpy.float64
    depth = np.asarray(depth)
    return depth


def read_rgb(path):
    img = Image.open(path).convert("RGB")

    # PIL to numpy
    img = np.array(img, dtype=np.float32, copy=False) / 255.0
    img = img[:370, :1220, :]  # crop image        

    return img


torch.set_grad_enabled(False)
@click.command()
@click.option('--bs', default=1, help='Batch size')
@click.option('--sequence_distance', default=10, help='total frames distance')
@click.option('--angle', default=10, help='experiment prefix')
@click.option('--step', default=0.5, help='experiment prefix')
@click.option('--max_distance', default=10.1, help='max pose sample distance')
@click.option('--frames_interval', default=0.4, help='Interval between supervision frames')
@click.option('--preprocess_root', default="", help='path to preprocess folder')
@click.option('--root', default="", help='path to dataset folder')
@click.option('--recon_save_dir', default="")
def main(
        root, preprocess_root,
        bs, recon_save_dir,
        sequence_distance,
        frames_interval, 
        angle, step, max_distance,
):

        
        data_module = KittiDataModule(
            root=root,
            n_rays=1000000, # Get all available lidar points
            preprocess_root=preprocess_root,
            sequence_distance=sequence_distance,
            n_sources=1000, # Get all frames in sequence
            frames_interval=frames_interval,
            batch_size=bs,
            num_workers=4,
        )
        data_module.setup_val_ds()
        data_loader = data_module.val_dataloader()
        
              
        rel_poses = sample_rel_poses(step=step, angle=angle, max_distance=max_distance)
        cnt = 0        
        for batch in tqdm(data_loader):
            cnt += 1

            for i in range(bs):
                frame_id = batch['frame_id'][i]
                sequence = batch['sequence'][i]

                tsdf_save_dir = os.path.join(recon_save_dir, "tsdf", sequence)        
                depth_save_dir = os.path.join(recon_save_dir, "depth", sequence)
                render_rgb_save_dir = os.path.join(recon_save_dir, "render_rgb", sequence)
                os.makedirs(tsdf_save_dir, exist_ok=True)                

                cam_K = batch["cam_K"][i].detach().cpu().numpy()
                T_velo2cam = batch["T_velo_2_cam"][i].detach().cpu().numpy()

               
                tsdf_save_path = os.path.join(tsdf_save_dir, frame_id + ".npy")
                if os.path.exists(tsdf_save_path):
                    print("Existed", tsdf_save_path)
                    continue

                
                scene_size = (51.2, 51.2, 6.4)
                vox_origin = np.array([0, -25.6, -2])
                vol_bnds = np.zeros((3,2))
                vol_bnds[:,0] = vox_origin
                vol_bnds[:,1] = vox_origin + np.array([scene_size[0], scene_size[1], scene_size[2]])

                tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.2)

                for (step, angle), rel_pose in tqdm(rel_poses.items()):
                    rel_pose = rel_pose.numpy()
                    depth_filepath = os.path.join(depth_save_dir, "{}_{}_{}.npy".format(frame_id, step, angle))
                    depth = np.load(depth_filepath)
                    
                    render_rgb_filepath = os.path.join(render_rgb_save_dir, "{}_{}_{}.png".format(frame_id, step, angle))
                    rgb = read_rgb(render_rgb_filepath) * 255.0

                    tsdf_vol.integrate(rgb, depth, cam_K, np.linalg.inv(T_velo2cam) @ rel_pose, obs_weight=1.)
                    
                
                tsdf_grid, _ = tsdf_vol.get_volume()
                verts, faces, norms, colors = tsdf_vol.get_mesh()
                
                
                np.save(tsdf_save_path, tsdf_grid)
                print("saved to", tsdf_save_path)

        
if __name__ == "__main__":
    main()
