from scenerf.data.semantic_kitti.kitti_dm import KittiDataModule
import numpy as np
from tqdm import tqdm
import os
import click
import torch
import imageio
from PIL import Image
from scenerf.loss.sscMetrics import SSCMetrics

def tsdf2occ(tsdf, th, max_th=4.0):
    occ = np.zeros(tsdf.shape)
    th_indivi = (0.1 + np.arange(256).reshape(256, 1, 1) * 0.2) * th
    th_indivi[th_indivi < 0.2] = 0.2
    th_indivi[th_indivi > max_th] = max_th
    occ[(np.abs(tsdf) < th_indivi) & (np.abs(tsdf) != 255)] = 1
    return occ
    

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
@click.option('--frames_interval', default=0.4, help='Interval between supervision frames')
@click.option('--preprocess_root', default="", help='path to preprocess folder')
@click.option('--root', default="", help='path to dataset folder')
@click.option('--recon_save_dir', default="")
def main(
        root, preprocess_root,
        bs, recon_save_dir,
        sequence_distance,
        frames_interval,
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
        
   
        metric = SSCMetrics(2) 
        fov_metric = SSCMetrics(2) 
        cnt = 0
        for batch in tqdm(data_loader):
            cnt += 1
            for i in range(bs):
                frame_id = batch['frame_id'][i]
                target_1_1 = batch['target_1_1'][i]
                sequence = batch['sequence'][i]
                fov_mask = batch['fov_mask_1'][i].reshape(target_1_1.shape)
                
                tsdf_save_dir = os.path.join(recon_save_dir, "tsdf", sequence)   
                tsdf_save_path = os.path.join(tsdf_save_dir, frame_id + ".npy")
                tsdf = np.load(tsdf_save_path)
                
                t = np.copy(target_1_1)
                t[target_1_1 == 255] = 0
                max_z = t.nonzero()[2].max()
                
                occ = tsdf2occ(tsdf, 0.25, 6.0)
                occ[:, :, max_z:] = 0 # don't evaluate points higher than the range of lidar
                
                metric.add_batch(occ, target_1_1)
                fov_metric.add_batch(occ, target_1_1, fov_mask)

                             
        print("========================")
        print("=========Summary========")   
        print("========================")
        stats = metric.get_stats()               
        print("==== Whole Scene ====")
        print(stats['iou'], stats['precision'], stats['recall'])

        fov_stats = fov_metric.get_stats()
        print("==== in FOV ====")
        print(fov_stats['iou'], fov_stats['precision'], fov_stats['recall'])

if __name__ == "__main__":
    main()
