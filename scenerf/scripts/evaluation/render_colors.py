import os
import shutil

import click
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from scenerf.models.scenerf import scenerf
from scenerf.data.semantic_kitti.kitti_dm import KittiDataModule


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


    with torch.no_grad():
        for batch in tqdm(data_loader):

            batch["cam_K"] = batch["cam_K"].cuda()
            batch["T_velo_2_cam"] = batch["T_velo_2_cam"].cuda()
     
            img_inputs = batch["img_inputs"].cuda()
            T_cam2velo = torch.inverse(batch["T_velo_2_cam"][0])
            cam_K = batch['cam_K'][0]
            inv_K = torch.inverse(cam_K)
      
            pix_coords, out_pix_coords, _ = model.spherical_mapping.from_pixels( inv_K=inv_K)
            x_rgbs = model.net_rgb(img_inputs, pix=pix_coords, pix_sphere=out_pix_coords)
           
            for i in range(bs):
                x_rgb = {}
                for k in x_rgbs:
                    x_rgb[k] = x_rgbs[k][i]

                frame_id = batch['frame_id'][i]
                sequence = batch['sequence'][i]
                
                cam_K = batch["cam_K"][i]

                source_distances = batch["source_distances"][i]
                source_frame_ids = batch['source_frame_ids'][i]
                img_sources = batch["img_sources"][i]
                T_source2infers = batch["T_source2infers"][i]
          
                for sid in tqdm(range(len(img_sources))):
                    # img_source = img_sources[sid]
                    T_source2infer = T_source2infers[sid]
                    source_distance = source_distances[sid]


                    source_frame_id = source_frame_ids[sid]

                    rgb_save_dir = os.path.join(eval_save_dir, "rgb", sequence)
                    render_rgb_save_dir = os.path.join(eval_save_dir, "render_rgb", sequence)

                    os.makedirs(rgb_save_dir, exist_ok=True)
                    os.makedirs(render_rgb_save_dir, exist_ok=True)

                   
                    rgb_filepath = os.path.join(rgb_save_dir,
                                                "{}_{}_{:.2f}.png".format(frame_id, source_frame_id, source_distance))
                    render_rgb_filepath = os.path.join(render_rgb_save_dir,
                                                "{}_{}_{:.2f}.png".format(frame_id, source_frame_id, source_distance))

                    if os.path.exists(render_rgb_filepath):
                        continue

                    # Save corresponding rgb image
                    if not os.path.exists(rgb_filepath):
                        source_path = os.path.join(
                            root,
                            "dataset/sequences/08/image_2/{}.png".format(source_frame_id))
                        shutil.copyfile(source_path, rgb_filepath)

                    img_size = (1220, 370)
                    xs = torch.arange(start=0, end=img_size[0], step=3).type_as(cam_K)
                    ys = torch.arange(start=0, end=img_size[1], step=3).type_as(cam_K)
                    grid_x, grid_y = torch.meshgrid(xs, ys)
                    rendered_im_size = grid_x.shape

                    sampled_pixels = torch.cat([
                        grid_x.unsqueeze(-1),
                        grid_y.unsqueeze(-1)
                    ], dim=2).reshape(-1, 2)


                    render_out_dict = model.render_rays_batch(cam_K,
                                                              T_source2infer.type_as(cam_K),
                                                              x_rgb,
                                                              T_cam2velo=T_cam2velo,
                                                              ray_batch_size=4000,
                                                              sampled_pixels=sampled_pixels)

                    depth_rendered = render_out_dict['depth']
                    color_rendered = render_out_dict['color']
                    
                    color_rendered_np = color_rendered.clamp(0, 1).detach().cpu().numpy()
                    color_rendered_np = color_rendered_np.reshape(rendered_im_size[0], rendered_im_size[1], 3)
                    color_rendered_np = np.transpose(color_rendered_np, (1, 0, 2))
                    plt.imsave(render_rgb_filepath, color_rendered_np)
                    print("Color saved {}".format(render_rgb_filepath))                

if __name__ == "__main__":
    main()
