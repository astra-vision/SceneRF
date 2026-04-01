from scenerf.data.bundlefusion.bundlefusion_dm import BundlefusionDM
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
@click.option('--eval_save_dir', default="")
def main(
    root, dataset,
    bs, n_gpus, n_workers_per_gpu,
    eval_save_dir
    ):

    data_module = BundlefusionDM(
        dataset=dataset,
        root=root,
        batch_size=int(bs / n_gpus),
        num_workers=int(n_workers_per_gpu),
    )
    data_module.setup_val_ds()
    data_loader = data_module.val_dataloader()

    cnt = 0

    agg_depth_errors = {}
    agg_n_frames = {}
    for batch in tqdm(data_loader):
        cnt += 1
       
        for i in range(bs):
            
            frame_id = batch['frame_id'][i]
            sequence = batch['sequence'][i]
            
            save_dir = os.path.join("{}/depth_metrics".format(eval_save_dir), sequence)

            save_filepath = os.path.join(save_dir,"{}.npy".format(frame_id))

            with open(save_filepath, "rb") as handle:
                data = pickle.load(handle)
            depth_errors = data["depth_errors"]
            n_frames = data["n_frames"]

            for k in depth_errors:  
                if k not in agg_depth_errors:
                    agg_depth_errors[k] = depth_errors[k]
                    agg_n_frames[k] = n_frames[k]
                else: 
                    agg_depth_errors[k] += depth_errors[k]
                    agg_n_frames[k] += n_frames[k]
            
            if cnt % 20 == 0:
                print("=================")
                print("==== batch {} ====".format(cnt))
                print("=================")
                print_metrics(agg_depth_errors, agg_n_frames)
    print("=================")
    print("====== End ======")
    print("=================")
    # print(model_name)
    print_metrics(agg_depth_errors, agg_n_frames)

            
            
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
