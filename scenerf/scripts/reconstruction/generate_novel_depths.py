import os
import PIL.Image as pil
import click
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F

from scenerf.models.scenerf import scenerf
from scenerf.data.semantic_kitti.kitti_dm import KittiDataModule
from scenerf.models.utils import depth2disp, sample_rel_poses


@click.command()
@click.option('--model_path', default="", help='path to checkpoint')
@click.option('--bs', default=1, help='batch size')
@click.option('--sequence_distance', default=10, help='total frames distance')
@click.option('--scale', default=2, help='total frames distance')
@click.option('--angle', default=10, help='experiment prefix')
@click.option('--step', default=0.5, help='experiment prefix')
@click.option('--max_distance', default=10.1, help='max pose sample distance')
@click.option('--frames_interval', default=0.4, help='Interval between supervision frames')
@click.option('--preprocess_root', default="", help='path to preprocess folder')
@click.option('--root', default="", help='path to dataset folder')
@click.option('--recon_save_dir', default="")
def main(
    root, preprocess_root, recon_save_dir,  model_path, bs,
    sequence_distance, frames_interval, scale,
    angle, step, max_distance):

    torch.set_grad_enabled(False)

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
         
            cam_K = batch['cam_K'][0]
            inv_K = torch.inverse(cam_K)


            pix_coords, out_pix_coords, _ = model.spherical_mapping.from_pixels(
                inv_K=inv_K)
            x_rgbs = model.net_rgb(img_inputs, pix=pix_coords,
                                pix_sphere=out_pix_coords)
            
            rel_poses = sample_rel_poses(step=step, angle=angle, max_distance=max_distance)

            for i in range(bs): 
                x_rgb = {}
                for k in x_rgbs:
                    x_rgb[k] = x_rgbs[k][i]
              
                frame_id = batch['frame_id'][i]
                sequence = batch['sequence'][i]
                
                
                cam_K = batch["cam_K"][i]
                for (step, angle), rel_pose in tqdm(rel_poses.items()):
                    
                    T_source2infer = rel_pose

                    depth_save_dir = os.path.join(recon_save_dir, "depth", sequence)
                    depth_visual_save_dir = os.path.join(recon_save_dir, "depth_visual", sequence)
                    render_rgb_save_dir = os.path.join(recon_save_dir, "render_rgb", sequence)

                    os.makedirs(depth_save_dir, exist_ok=True)
                    os.makedirs(depth_visual_save_dir, exist_ok=True)
                    os.makedirs(render_rgb_save_dir, exist_ok=True)

                    depth_visual_filepath = os.path.join(depth_visual_save_dir, "{}_{}_{}.png".format(frame_id, step, angle))
                    depth_filepath = os.path.join(depth_save_dir, "{}_{}_{}.npy".format(frame_id, step, angle))

                    render_rgb_filepath = os.path.join(render_rgb_save_dir,
                                                "{}_{}_{}.png".format(frame_id, step, angle))
                    print(depth_visual_filepath)
                    if os.path.exists(depth_filepath) and os.path.exists(depth_visual_filepath) and os.path.exists(render_rgb_filepath):
                        print("existed")
                        continue
                    
                    img_size = (1220, 370)
                    xs = torch.arange(start=0, end=img_size[0], step=scale).type_as(cam_K)
                    ys = torch.arange(start=0, end=img_size[1], step=scale).type_as(cam_K)
                    grid_x, grid_y = torch.meshgrid(xs, ys)
                    rendered_im_size = grid_x.shape

                    sampled_pixels = torch.cat([
                        grid_x.unsqueeze(-1),
                        grid_y.unsqueeze(-1)
                    ], dim=2).reshape(-1, 2)


                    # for ci in range(0, sampled_pixels.shape[0], chunk):
                    render_out_dict = model.render_rays_batch(cam_K,
                                                            T_source2infer.type_as(cam_K),
                                                            x_rgb,
                                                            #   T_cam2velo=None,              
                                                            #   depth_window=100,
                                                            ray_batch_size=5000,
                                                            sampled_pixels=sampled_pixels)

                    depth_rendered = render_out_dict['depth'].reshape(rendered_im_size[0], rendered_im_size[1])
                    color_rendered = render_out_dict['color'].reshape(rendered_im_size[0], rendered_im_size[1], 3)
      
                    if scale != 1:
                        depth_rendered = F.interpolate(
                            depth_rendered.T.unsqueeze(0).unsqueeze(0) ,
                            size=(370, 1220),
                            mode="bilinear"
                        )
                        color_rendered = F.interpolate(
                            color_rendered.permute(2, 1, 0).unsqueeze(0) ,
                            size=(370, 1220),
                            mode="bilinear"
                        )
                    else:
                        depth_rendered = depth_rendered.T.unsqueeze(0).unsqueeze(0) 
                        color_rendered = color_rendered.permute(2, 1, 0).unsqueeze(0)
              
                    
                  
                    
                    color_rendered_np = color_rendered.clamp(0, 1).squeeze().permute(2, 1, 0).detach().cpu().numpy()
                    color_rendered_np = np.transpose(color_rendered_np, (1, 0, 2))
                    plt.imsave(render_rgb_filepath, color_rendered_np)             

                    
                    depth_rendered = depth_rendered.squeeze()
                    depth_rendered_np = depth_rendered.detach().cpu().numpy()
                    np.save(depth_filepath, depth_rendered_np)
                    print("saved depth", depth_filepath)
                  
                    disp = depth2disp(depth_rendered)
                    disp_resized = disp
                    disp_resized = disp_resized.squeeze()
                    disp_np = disp_resized.detach().cpu().numpy()
                    

                    # Save for visualizing
                    vmax = np.percentile(disp_np, 95)
                    vmin = disp_np.min()
                    normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
                    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
                    colormapped_im = (mapper.to_rgba(disp_np)[:, :, :3] * 255).astype(np.uint8)
                    im = pil.fromarray(colormapped_im)
                    im.save(depth_visual_filepath)

if __name__ == "__main__":
    main()
