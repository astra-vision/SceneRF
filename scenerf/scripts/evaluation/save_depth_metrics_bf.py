from scenerf.models.scenerf_bf import SceneRF
from scenerf.data.bundlefusion.bundlefusion_dm import BundlefusionDM
import torch
import numpy as np
import os
from tqdm import tqdm
import PIL.Image as pil
import click
import pickle
import math
from scenerf.loss.depth_metrics import compute_depth_errors
import torch.nn.functional as F


torch.set_grad_enabled(False)

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
@click.option('--root', default="", help='path to dataset folder')

@click.option('--model_path', default="", help='model path')
@click.option('--eval_save_dir', default="")

def main(
    root, dataset,
    bs, n_gpus, n_workers_per_gpu,
    model_path, eval_save_dir):


    data_module = BundlefusionDM(
        dataset=dataset,
        root=root,
        batch_size=int(bs / n_gpus),
        num_workers=int(n_workers_per_gpu),
        n_sources=1000,
    )
    data_module.setup_val_ds()
    data_loader = data_module.val_dataloader(shuffle=True)

    model = SceneRF.load_from_checkpoint(model_path)


    model.cuda()
    model.eval()

    cnt = 0
    
    for batch in tqdm(data_loader):
        cnt += 1
        img_input = batch["img_inputs"].cuda()

        cam_K = batch['cam_K_depth'][0].cuda()
        inv_K = torch.inverse(cam_K)
        pix_coords, out_pix_coords, _ = model.spherical_mapping.from_pixels(
            inv_K=inv_K)
        x_rgbs = model.net_rgb(img_input, pix=pix_coords, pix_sphere=out_pix_coords)


        for i in range(bs):
            x_rgb = {}
            for k in x_rgbs:
                x_rgb[k] = x_rgbs[k][i]

            frame_id = batch['frame_id'][i]
            sequence = batch['sequence'][i]
            source_depths = batch['source_depths'][i]

            save_dir = os.path.join("{}/depth_metrics".format(eval_save_dir), sequence)
            os.makedirs(save_dir, exist_ok=True)
            save_filepath = os.path.join(save_dir,"{}.npy".format(frame_id))
            if os.path.exists(save_filepath):
                continue
            
            
            source_frame_ids = batch['source_frame_ids'][i]
            img_sources = batch["img_sources"][i]
            T_source2infers = batch["T_source2infers"][i]

            agg_depth_errors = {}
            n_frames = {}

            for source_id in tqdm(range(len(img_sources))):
                
                T_source2infer = T_source2infers[source_id].cuda()
                source_frame_id = source_frame_ids[source_id]
                source_distance = abs(int(source_frame_id) - int(frame_id))
                source_depth = torch.from_numpy(source_depths[source_id]).cuda()
                nonzero_indices = torch.nonzero(source_depth)
                nonzero_indices[:, [0, 1]] = nonzero_indices[:, [1, 0]]
                # Evaluate at half scale
                nonzero_indices = nonzero_indices[(nonzero_indices[:, 0] % 2 == 0) & (nonzero_indices[:, 0] % 2 == 0)]
                gt_depth_source = source_depth[nonzero_indices[:, 1], nonzero_indices[:, 0]]
                sample_pixels = nonzero_indices.float()
                
                
                render_out_dict = model.render_rays_batch(
                    cam_K,
                    T_source2infer,
                    x_rgb,
                    ray_batch_size=min(8000, sample_pixels.shape[0]),
                    sampled_pixels=sample_pixels)
                
                pred_depth_source = render_out_dict['depth']
                
                gt_depth_source = gt_depth_source.clamp(0.1, 10.0)
                pred_depth_source = pred_depth_source.clamp(0.1, 10.0)
                depth_errors = evaluate_depth(gt_depth_source, pred_depth_source)
                
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
