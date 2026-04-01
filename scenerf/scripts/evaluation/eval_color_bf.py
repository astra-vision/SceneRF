import torch
import lpips
import os
import glob
from torchvision import transforms
from PIL import Image
import numpy as np
import skimage.metrics
from tqdm import tqdm
from collections import defaultdict
from skimage.transform import resize
import click

torch.set_grad_enabled(False)

lpips_func = lpips.LPIPS(net="vgg").cuda()

def compute_psnr(rgb, gt):
    psnr = skimage.metrics.peak_signal_noise_ratio(rgb, gt, data_range=1)
    return psnr

def compute_lpips(img1, img2):    
    lpips_func.requires_grad = False
    img1 = (img1 - 0.5) * 2
    img2 = (img2 - 0.5) * 2
    return lpips_func(img1, img2)

def compute_ssim(rgb, gt):
    ssim = skimage.metrics.structural_similarity(rgb, gt, multichannel=True, data_range=1)
    return ssim

def print_metrics(psnr_accum, ssim_accum, lpips_accum, cnt_accum):
    print("|distance |psnr |ssim   |lpips     |n_frames|")

    total_psnr = 0
    total_ssim = 0
    total_lpips = 0
    total_frame = 0 
    for distance in sorted(psnr_accum):
        total_psnr +=  psnr_accum[distance]
        total_ssim += ssim_accum[distance]
        total_lpips += lpips_accum[distance]
        total_frame += cnt_accum[distance]
        
        print("|{}|{:02.6f}|{:.6f}|{:.6f}|{:.6f}|".format(
            distance,
            psnr_accum[distance]/cnt_accum[distance],
            ssim_accum[distance]/cnt_accum[distance],
            lpips_accum[distance]/cnt_accum[distance],
            cnt_accum[distance]
        ))
        
    print("|{}|{:02.6f}|{:.6f}|{:.6f}|{:.6f}|".format(
            "All     ",
            total_psnr/total_frame,
            total_ssim/total_frame,
            total_lpips/total_frame,
            total_frame
        ))

@click.command()
@click.option('--eval_save_dir', default="")
@click.option('--dataset', default='bf', help='bf or tum_rgbd dataset to evaluate on')
def main(eval_save_dir, dataset):

    if dataset == "bf":
        sequence = "copyroom"
    elif dataset == "tum_rgbd":
        sequence = "rgbd_dataset_freiburg3_long_office_household"
    rgb_save_dir = os.path.join(eval_save_dir, "rgb", sequence)
    render_rgb_save_dir = os.path.join(eval_save_dir, "render_rgb", sequence)
    rgb_paths = glob.glob(os.path.join(rgb_save_dir, "*.png"))

    # rendered_rgb_paths = glob.glob(os.path.join(render_rgb_save_dir, "*.png"))
    psnr_accum = defaultdict(float)
    ssim_accum = defaultdict(float)
    lpips_accum = defaultdict(float)
    cnt_accum = defaultdict(int)
    n = 0
    for i in tqdm(range(len(rgb_paths))):
        rgb_path = rgb_paths[i]
        print(rgb_path)
        filename = os.path.basename(rgb_path)
        frame_id, _, source_distance = filename[:-4].split("_")
        
        
        rendered_rgb_path = os.path.join(render_rgb_save_dir, filename)
        
        rgb = Image.open(rgb_path).convert("RGB")
        rgb = np.array(rgb, dtype=np.float32, copy=False) / 255.0

            
        rendered_rgb = Image.open(rendered_rgb_path).convert("RGB")
    
        rendered_rgb = np.array(rendered_rgb, dtype=np.float32, copy=False) / 255.0
        

        psnr_score = compute_psnr(rendered_rgb, rgb)
        ssim_score = compute_ssim(rendered_rgb, rgb)

        trans = transforms.ToTensor()
        rgb = trans(rgb).unsqueeze(0).cuda()
        rendered_rgb = trans(rendered_rgb).unsqueeze(0).cuda()
        lpips_score = compute_lpips(rendered_rgb, rgb).mean()
        

        k = source_distance
        # k = 0
        psnr_accum[k] +=  psnr_score
        ssim_accum[k] += ssim_score
        lpips_accum[k] += lpips_score.item()
        cnt_accum[k] += 1
        
        
        n += 1
        if n % 100 == 0:
            print("====> Step ", n)
            print_metrics(psnr_accum, ssim_accum, lpips_accum, cnt_accum)

    print("====> End")
    print_metrics(psnr_accum, ssim_accum, lpips_accum, cnt_accum)
    
if __name__ == "__main__":
    main()
