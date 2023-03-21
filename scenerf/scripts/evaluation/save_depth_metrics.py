from scenerf.models.scenerf import scenerf
from scenerf.data.semantic_kitti.kitti_dm import KittiDataModule
import torch
import numpy as np
import os
from tqdm import tqdm

import click
import pickle
import math
from scenerf.loss.depth_metrics import compute_depth_errors

torch.set_grad_enabled(False)

def evaluate_depth(gt_depth, pred_depth):
    depth_errors = []

    
    depth_error = compute_depth_errors(
        gt=gt_depth.reshape(-1).detach().cpu().numpy(),
        pred=pred_depth.reshape(-1).detach().cpu().numpy(),
    )
    # print(depth_error)
    depth_errors.append(depth_error)

    
    agg_depth_errors = np.array(depth_errors).sum(0)

    return agg_depth_errors



@click.command()
@click.option('--model_path', default="", help='path to checkpoint')
@click.option('--bs', default=1, help='batch size')
@click.option('--sequence_distance', default=10, help='total frames distance')
@click.option('--frames_interval', default=0.4, help='Interval between supervision frames')
@click.option('--preprocess_root', default="", help='path to preprocess folder')
@click.option('--eval_save_dir', default="", help='Folder for saving intermediate data')
@click.option('--root', default="", help='path to dataset folder')
def main(
    root, preprocess_root, eval_save_dir,  model_path, bs,
    sequence_distance, frames_interval):
  
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
    model = scenerf.load_from_checkpoint(model_path)
    model.cuda()
    model.eval()
    
    
    cnt = 0
    for batch in tqdm(data_loader):

        cnt += 1
        img_input = batch["img_inputs"].cuda()

        cam_K = batch['cam_K'][0].cuda()
        inv_K = torch.inverse(cam_K)
        batch["T_velo_2_cam"] = batch["T_velo_2_cam"].cuda()

        pix_coords, out_pix_coords, _ = model.spherical_mapping.from_pixels(
            inv_K=inv_K)
        x_rgbs = model.net_rgb(img_input, pix=pix_coords, pix_sphere=out_pix_coords)
        
        for i in range(bs):

            x_rgb = {}
            for k in x_rgbs:
                x_rgb[k] = x_rgbs[k][i]
          
            frame_id = batch['frame_id'][i]
            sequence = batch['sequence'][i]

            save_dir = os.path.join(eval_save_dir, "depth_metrics", sequence)
            os.makedirs(save_dir, exist_ok=True)
            save_filepath = os.path.join(save_dir,"{}.npy".format(frame_id))
            if os.path.exists(save_filepath):
                continue
            
            source_distances = batch["source_distances"][i]
            source_frame_ids = batch['source_frame_ids'][i]
            img_sources = batch["img_sources"][i]
            T_source2infers = batch["T_source2infers"][i]
            loc2d_with_depths = batch['loc2d_with_depths'][i]
            lidar_depths = batch['lidar_depths'][i]

            agg_depth_errors = {}
            n_frames = {}

            for source_id in tqdm(range(len(img_sources))):
                
                T_source2infer = T_source2infers[source_id].cuda()
                source_distance = source_distances[source_id]
                loc2d_with_depth = loc2d_with_depths[source_id].cuda()
                lidar_depth = lidar_depths[source_id].cuda()

                
                gt_sampled_pixels_infer = loc2d_with_depth.float()
                gt_depth_infer = lidar_depth
                
                
                render_out_dict = model.render_rays_batch(
                    cam_K,
                    T_source2infer,
                    x_rgb,
                    ray_batch_size=min(4000, gt_sampled_pixels_infer.shape[0]),
                    sampled_pixels=gt_sampled_pixels_infer)
                
                pred_depth_infer = render_out_dict['depth']
                
                depth_errors = evaluate_depth(gt_depth_infer, pred_depth_infer)
                
                k = math.ceil(source_distance)
                
                if k not in agg_depth_errors:
                    agg_depth_errors[k] = depth_errors
                    n_frames[k] = 1
                else: 
                    agg_depth_errors[k] += depth_errors
                    n_frames[k] += 1

            out_dict = {
                "depth_errors": agg_depth_errors,
                "n_frames": n_frames
            }
            with open(save_filepath, "wb") as output_file:
                pickle.dump(out_dict, output_file)
                print("Saved to", save_filepath)

            
            print("=================")
            print("==== Frame {} ====".format(frame_id))
            print("=================")
            print_metrics(agg_depth_errors, n_frames)

            
            
def print_metrics(agg_depth_errors, n_frames):
    print("|distance|abs_rel |sq_rel  |rmse     |rmse_log|a1      |a2      |a3      |n_frames|")

    total_depth_errors = None
    total_frame = 0 
    for distance in sorted(agg_depth_errors):
        if total_depth_errors is None:
            total_depth_errors = np.copy(agg_depth_errors[distance])
        else:
            total_depth_errors = total_depth_errors + agg_depth_errors[distance]
        metric_list = ["abs_rel", "sq_rel",
                        "rmse", "rmse_log", "a1", "a2", "a3"]
        print("|{:08d}|{:02.6f}|{:.6f}|{:.6f}|{:.6f}|{:.6f}|{:.6f}|{:.6f}|{:08d}|".format(
            distance,
            agg_depth_errors[distance][0]/n_frames[distance],
            agg_depth_errors[distance][1]/n_frames[distance],
            agg_depth_errors[distance][2]/n_frames[distance],
            agg_depth_errors[distance][3]/n_frames[distance],
            agg_depth_errors[distance][4]/n_frames[distance],
            agg_depth_errors[distance][5]/n_frames[distance],
            agg_depth_errors[distance][6]/n_frames[distance],
            n_frames[distance]
        ))
        total_frame += n_frames[distance]
    print("|{}|{:02.6f}|{:.6f}|{:.6f}|{:.6f}|{:.6f}|{:.6f}|{:.6f}|{:08d}|".format(
            "All     ",
            total_depth_errors[0]/total_frame,
            total_depth_errors[1]/total_frame,
            total_depth_errors[2]/total_frame,
            total_depth_errors[3]/total_frame,
            total_depth_errors[4]/total_frame,
            total_depth_errors[5]/total_frame,
            total_depth_errors[6]/total_frame,
            total_frame
        ))

if __name__ == "__main__":
    main()
