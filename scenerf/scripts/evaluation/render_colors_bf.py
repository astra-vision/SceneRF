import os
import shutil

from PIL import Image
import PIL.Image as pil
import click
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F

from scenerf.data.bundlefusion.bundlefusion_dm import BundlefusionDM

from scenerf.models.scenerf_bf import SceneRF

from scenerf.models.utils import depth2disp


def disparity_normalization_vis(disparity):
    """
    :param disparity: Bx1xHxW, pytorch tensor of float32
    :return:
    """
    assert len(disparity.size()) == 4 and disparity.size(1) == 1
    disp_min = torch.amin(disparity, (1, 2, 3), keepdim=True)
    disp_max = torch.amax(disparity, (1, 2, 3), keepdim=True)
    disparity_syn_scaled = (disparity - disp_min) / (disp_max - disp_min)
    disparity_syn_scaled = torch.clip(disparity_syn_scaled, 0.0, 1.0)
    return disparity_syn_scaled


@click.command()
@click.option('--n_gpus', default=1, help='number of GPUs')
@click.option('--bs', default=1, help='Batch size')
@click.option('--save_depth', default=True)
@click.option('--model_path', default="", help='model path')
@click.option('--n_workers_per_gpu', default=10, help='number of workers per GPU')
@click.option('--dataset', default='bf', help='bf or tum_rgbd dataset to eval on')
@click.option('--root', default="/gpfsdswork/dataset/bundlefusion", help='path to dataset folder')
@click.option('--eval_save_dir', default="")
def main(
        root, dataset, bs, n_gpus, n_workers_per_gpu,
        model_path, save_depth, eval_save_dir
):
    torch.set_grad_enabled(False)

    data_module = BundlefusionDM(
        dataset=dataset,
        root=root,
        batch_size=int(bs / n_gpus),
        num_workers=int(n_workers_per_gpu),
        n_sources=1000,
    )
    data_module.setup_val_ds()
    val_dataloader = data_module.val_dataloader(shuffle=True)


    model = SceneRF.load_from_checkpoint(model_path)
   
    model.cuda()
    model.eval()

    with torch.no_grad():
        for batch in tqdm(val_dataloader):
        
            cam_K = batch['cam_K_depth'][0].cuda()
     
            img_input = batch["img_inputs"].cuda()
            
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
                source_frame_ids = batch['source_frame_ids'][i]
                
                source_frame_ids = batch['source_frame_ids'][i]
                img_sources = batch["img_sources"][i]
                T_source2infers = batch["T_source2infers"][i]
                for sid in tqdm(range(len(img_sources))):
                    T_source2infer = T_source2infers[sid]
                    source_frame_id = source_frame_ids[sid]
                    source_distance = abs(int(source_frame_id) - int(frame_id))

                    depth_save_dir = os.path.join(eval_save_dir, "depth", sequence)
                    rgb_save_dir = os.path.join(eval_save_dir, "rgb", sequence)
                    depth_visual_save_dir = os.path.join(eval_save_dir, "depth_visual", sequence)
                    render_rgb_save_dir = os.path.join(eval_save_dir, "render_rgb", sequence)


                    os.makedirs(depth_save_dir, exist_ok=True)
                    os.makedirs(rgb_save_dir, exist_ok=True)
                    os.makedirs(depth_visual_save_dir, exist_ok=True)
                    os.makedirs(render_rgb_save_dir, exist_ok=True)

                    depth_visual_filepath = os.path.join(depth_visual_save_dir,
                                                         "{}_{}_{:.2f}.png".format(frame_id, source_frame_id,
                                                                                  source_distance))
                    depth_filepath = os.path.join(depth_save_dir,
                                                  "{}_{}_{:.2f}.png".format(frame_id, source_frame_id, source_distance))
                    rgb_filepath = os.path.join(rgb_save_dir,
                                                "{}_{}_{:.2f}.png".format(frame_id, source_frame_id, source_distance))
                    render_rgb_filepath = os.path.join(render_rgb_save_dir,
                                                "{}_{}_{:.2f}.png".format(frame_id, source_frame_id, source_distance))

                    
                    if os.path.exists(render_rgb_filepath):
                        continue

                    # Save corresponding rgb image
                    if not os.path.exists(rgb_filepath):
                        source_path = os.path.join(root, sequence, "frame-{}.color.jpg".format(source_frame_id))
                        shutil.copyfile(source_path, rgb_filepath)


                    img_size = (640, 480)
                    scale = 2
                    xs = torch.arange(start=0, end=img_size[0], step=scale).type_as(cam_K)
                    ys = torch.arange(start=0, end=img_size[1], step=scale).type_as(cam_K)
                    grid_x, grid_y = torch.meshgrid(xs, ys)
                    rendered_im_size = grid_x.shape

                    sampled_pixels = torch.cat([
                        grid_x.unsqueeze(-1),
                        grid_y.unsqueeze(-1)
                    ], dim=2).reshape(-1, 2)


                    render_out_dict = model.render_rays_batch(cam_K,
                                                            T_source2infer.type_as(cam_K),
                                                            x_rgb,
                                                            ray_batch_size=8000,
                                                            sampled_pixels=sampled_pixels)
                    
                    depth_rendered = render_out_dict['depth'].reshape(rendered_im_size[0], rendered_im_size[1])
                    color_rendered = render_out_dict['color'].reshape(rendered_im_size[0], rendered_im_size[1], 3)

                    depth_rendered = F.interpolate(
                        depth_rendered.T.unsqueeze(0).unsqueeze(0) ,
                        scale_factor=scale,
                        mode="bilinear"
                    )
                    color_rendered = F.interpolate(
                        color_rendered.permute(2, 1, 0).unsqueeze(0) ,
                        scale_factor=scale,
                        mode="bilinear"
                    )

                    color_rendered_np = color_rendered.clamp(0, 1).detach().cpu().numpy().squeeze()

                    color_rendered_np = np.transpose(color_rendered_np, (1, 2, 0))

                    plt.imsave(render_rgb_filepath, color_rendered_np)
                    print("Color saved {}".format(render_rgb_filepath))                

                    if save_depth:
                        
                        disp = depth2disp(depth_rendered, min_depth=0.1, max_depth=12.0).squeeze()
                        disp_resized = disp
                        disp_np = disp_resized.detach().cpu().numpy()
                        img = Image.fromarray((disp_np * 255.0).astype(np.uint8))
                        img.save(depth_filepath)
                        print("saved depth", depth_filepath)

                        # Save for visualizing
                        # vmax = np.percentile(disp_np, 97)
                        vmax = disp_np.max()
                        # print(disp_np.min(), disp_np.max(), vmax)
                        # vmin = np.percentile(disp_np, 3)
                        vmin = disp_np.min()
                        
                        normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
                        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
                        # pdb.set_trace()
                        colormapped_im = (mapper.to_rgba(disp_np)[:, :, :3] * 255).astype(np.uint8)
                        im = pil.fromarray(colormapped_im)

                        im.save(depth_visual_filepath)
                        print("saved depth visual", depth_visual_filepath)
                    



if __name__ == "__main__":
    main()
