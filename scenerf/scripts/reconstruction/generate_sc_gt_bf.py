from scenerf.data.bundlefusion.bundlefusion_dm import BundlefusionDM
from scenerf.data.utils import fusion
import torch
import numpy as np
import os
from tqdm import tqdm
import click
import pickle


torch.set_grad_enabled(False)


@click.command()
@click.option('--n_gpus', default=1, help='number of GPUs')
@click.option('--bs', default=1, help='Batch size')
@click.option('--n_workers_per_gpu', default=3, help='number of workers per GPU')
@click.option('--dataset', default='bf', help='bf or tum_rgbd dataset to eval on')
@click.option('--root', default="", help='path to dataset folder')
@click.option('--recon_save_dir')
def main(root, dataset, bs, n_gpus, n_workers_per_gpu, recon_save_dir):

    data_module = BundlefusionDM(
        dataset,
        root=root,
        batch_size=int(bs / n_gpus),
        num_workers=int(n_workers_per_gpu),
        n_sources=1000,
        train_n_frames=16,
        val_n_frames=16,
        val_frame_interval=2
    )
    data_module.setup()
    data_loader = data_module.val_dataloader(shuffle=False)

    for batch in tqdm(data_loader):

        for i in range(bs):
            cam_K = batch['cam_K_depth'][i].numpy()
            inv_K = np.linalg.inv(cam_K)
            # inv_K = torch.inverse(cam_K)
            frame_id = batch['frame_id'][i]
           
            sequence = batch['sequence'][i]
            infer_depth = batch['infer_depths'][i].numpy()
            
            
            save_dir = os.path.join(recon_save_dir, "sc_gt")
            save_dir = os.path.join(save_dir, sequence)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_filepath = os.path.join(save_dir, "{}.pkl".format(frame_id))
            if os.path.exists(save_filepath):
                print("exist", save_filepath)
                continue

            source_depths = batch['source_depths'][i]
            img_sources = batch["img_sources"][i]
            T_source2infers = batch["T_source2infers"][i]


           
            voxel_size = 0.04
            sx, sy, sz = 4.8, 4.8, 3.84
            scene_size = (sx, sy, sz)
            vox_origin = (-sx / 2, -sy / 2, 0)
            vol_bnds = np.zeros((3,2))
            vol_bnds[:,0] = vox_origin
            vol_bnds[:,1] = vox_origin + np.array([scene_size[0], scene_size[1], scene_size[2]])

            tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=voxel_size, trunc_margin=10)
            
            for source_id in tqdm(range(len(img_sources))):
                
                T_source2infer = T_source2infers[source_id].numpy()
                source_depth = source_depths[source_id]
                img_source = img_sources[source_id]

                source_depth = torch.from_numpy(source_depth).unsqueeze(0).unsqueeze(0)
                source_depth = torch.nn.functional.interpolate(source_depth, size=(480, 640), mode='bilinear', align_corners=False).squeeze()
               
                rgb = img_source.permute(1, 2, 0).numpy() * 255

                tsdf_vol.integrate(rgb, source_depth, cam_K, T_source2infer, obs_weight=1.)
                
           
            verts, faces, norms, colors = tsdf_vol.get_mesh()
            tsdf_grid, _ = tsdf_vol.get_volume() 

            occ = np.zeros_like(tsdf_grid) + 255
            occ[(tsdf_grid > voxel_size)& (tsdf_grid != 255)] = 0 # the unknown voxels has tsdf value of 255
            occ[(abs(tsdf_grid) < voxel_size) & (tsdf_grid != 255)] = 1
            
            data = {
                    "tsdf_grid": tsdf_grid,
                    "occ": occ.astype(np.uint8),
                }
                
            with open(save_filepath, "wb") as handle:
                pickle.dump(data, handle)
                print(data['tsdf_grid'].shape)
                print("wrote to", save_filepath)


if __name__ == "__main__":
    main()
