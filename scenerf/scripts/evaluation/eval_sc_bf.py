from scenerf.data.bundlefusion.bundlefusion_dm import BundlefusionDM
import torch
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
import click
from scenerf.loss.sscMetrics import SSCMetrics
import pickle
from scenerf.loss.depth_metrics import compute_depth_errors

torch.set_grad_enabled(False)


def tsdf2occ(tsdf, min_th, th=0.25, max_th=0.2, voxel_size=0.04):
    occ = np.zeros(tsdf.shape)
    th_indivi = (voxel_size + np.arange(96).reshape(1, 1, 96) * voxel_size * th) 
    th_indivi[th_indivi < min_th] = min_th
    th_indivi[th_indivi > max_th] = max_th
    occ[(np.abs(tsdf) < th_indivi) & (np.abs(tsdf) != 255)] = 1
    return occ


def read_rgb(path):
    img = Image.open(path).convert("RGB")

    # PIL to numpy
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


metrics = SSCMetrics(2)

@click.command()
@click.option('--n_gpus', default=1, help='number of GPUs')
@click.option('--model_name', default="", help='model name')
@click.option('--bs', default=1, help='Batch size')
@click.option('--n_workers_per_gpu', default=3, help='number of workers per GPU')
@click.option('--dataset', default='bf', help='bf or tum_rgbd dataset to eval on')
@click.option('--root', default="/gpfsdswork/dataset/bundlefusion", help='path to dataset folder')
@click.option('--recon_save_dir')
def main(root, dataset, bs, n_gpus, n_workers_per_gpu, model_name, recon_save_dir):
    

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
    
    prefix = model_name
    target_dir = "/gpfswork/rech/trg/uyl37fq/preprocess_data/bundlefusion/sc_gt" 
    for batch in tqdm(data_loader):
        cnt += 1
        

        for i in range(bs):
            cam_K = batch['cam_K_depth'][i].numpy()
            inv_K = np.linalg.inv(cam_K)
            
            frame_id = batch['frame_id'][i]
           
            sequence = batch['sequence'][i]

            gt_save_dir = os.path.join(recon_save_dir, "sc_gt", sequence)
            gt_save_filepath = os.path.join(gt_save_dir, "{}.pkl".format(frame_id))

            with open(gt_save_filepath, "rb") as f:
                data = pickle.load(f)
                target = data["occ"]
            
            tsdf_save_dir = os.path.join(recon_save_dir, "tsdf", sequence)
            os.makedirs(tsdf_save_dir, exist_ok=True)
            tsdf_save_filepath = os.path.join(tsdf_save_dir, "{}.pkl".format(frame_id))


            with open(tsdf_save_filepath, "rb") as f:
                data = pickle.load(f)
                tsdf_grid = data["tsdf_grid"]

            voxel_size = 0.04
            occ = tsdf2occ(tsdf_grid, 
                th=0.1, 
                min_th=voxel_size,
                max_th=voxel_size * 10, 
                voxel_size=voxel_size)
            
            metrics.add_batch(occ, target)
            if cnt % 20 == 0:
                stats = metrics.get_stats()
                print("=====================================")
                print(stats['iou'], stats['precision'], stats['recall'])
            
            

if __name__ == "__main__":
    main()
