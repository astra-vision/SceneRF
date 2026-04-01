import logging
import os

import click
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from scenerf.data.bundlefusion.bundlefusion_dm import BundlefusionDM

from scenerf.models.scenerf_bf import SceneRF

seed_everything(42)

logger = logging.getLogger(__name__)




@click.command()
@click.option('--dataset', default='bf', help='bf or tum_rgbd dataset to train on')
@click.option('--logdir', default='', help='log directory')
@click.option('--root', default='', help='path to dataset folder')
@click.option('--bs', default=1, help='Batch size')
@click.option('--lr', default=1e-5, help='learning rate')
@click.option('--wd', default=0, help='weight decay')
@click.option('--n_gpus', default=1, help='number of GPUs')
@click.option('--pretrained_exp_name', default=None, help='pretrained exp_name')
@click.option('--n_workers_per_gpu', default=4, help='number of workers per GPU')
@click.option('--enable_log', default=False, help='enable log')
@click.option('--exp_prefix', default="exp", help='experiment prefix')
@click.option('--n_rays', default=1080, help='Total number of rays')
@click.option('--sample_grid_size', default=1, help='sample pixel stride')
@click.option('--smooth_loss_weight', default=0.0, help='smooth loss weight')

@click.option('--max_sample_depth', default=12, help='maximum sample depth')
@click.option('--eval_depth', default=10, help='cap depth at 10m for evaluation')

@click.option('--n_pts_per_gaussian', default=8, help='number of points sampled for each gaussian')
@click.option('--n_gaussians', default=4, help='number of gaussians')
@click.option('--n_pts_uni', default=32, help='number of points sampled uniformly')
@click.option('--n_pts_hier', default=32, help='number of points sampled hierarchically')
@click.option('--std', default=0.1, help='std of each gaussian')

@click.option('--add_fov_hor', default=14, help='angle added to left and right of the horizontal FOV')
@click.option('--add_fov_ver', default=11, help='angle added to top and bottom of the vertical FOV')
@click.option('--sphere_h', default=720, help='total frames distance')
@click.option('--sphere_w', default=960, help='total frames distance')

@click.option('--sampling_method', default="uniform", help='point sampling method')
@click.option('--som_sigma', default=0.02, help='sigma parameter for SOM')
@click.option('--net_2d', default="b7", help='')

@click.option('--max_epochs', default=30, help='max training epochs')
@click.option('--use_color', default=True, help='use color loss')
@click.option('--use_reprojection', default=True, help='use reprojection loss')

@click.option('--n_frames', default=16, help='number of frames in a sequence')
@click.option('--frame_interval', default=2, help='interval between frames in a sequence')

def main(
        dataset, root,
        bs, n_gpus, n_workers_per_gpu,
        exp_prefix, pretrained_exp_name,
        logdir, enable_log,
        lr, wd,
        n_rays, sample_grid_size,
        smooth_loss_weight,
        max_sample_depth, eval_depth,
        n_pts_uni, n_pts_hier,
        n_pts_per_gaussian, n_gaussians, std, som_sigma,
        add_fov_hor, add_fov_ver,
        use_color, use_reprojection,
        sphere_w, sphere_h, max_epochs,
        sampling_method, net_2d,
        n_frames, frame_interval):
    assert root != "" and os.path.isdir(root), "$BF_ROOT is not set"
    assert logdir != "" and os.path.isdir(logdir), "$BF_LOG is not set"
    exp_name = exp_prefix
    exp_name += "_lr{}_{}rays_{}".format(lr, n_rays, net_2d)
    exp_name += "_nGaus{}_nPtsPerGaus{}_std{}_SOMSigma{}".format(n_gaussians, n_pts_per_gaussian, std, som_sigma)
    
    exp_name += "_sphere{}x{}_addfov{}x{}".format(sphere_w, sphere_h, add_fov_hor, add_fov_ver)
    exp_name += "_nFrames{}_frameInterval{}".format(n_frames, frame_interval)
    
    if not use_reprojection:
        exp_name += "NoReproj"
    if not use_color:
        exp_name += "NoColor"


    # Setup dataloaders
    # max_epochs = 20

    data_module = BundlefusionDM(
        dataset=dataset,
        root=root,
        batch_size=int(bs / n_gpus),
        num_workers=int(n_workers_per_gpu),
    )

    print(exp_name)
    if pretrained_exp_name is not None:
        exp_name = pretrained_exp_name


    # Initialize MonoScene model
    model = SceneRF(
        lr=lr,
        n_pts_uni=n_pts_uni,
        n_pts_hier=n_pts_hier,
        weight_decay=wd,
        n_rays=n_rays,
        smooth_loss_weight=smooth_loss_weight,
        sample_grid_size=sample_grid_size,
        max_sample_depth=max_sample_depth,
        sampling_method=sampling_method,
        n_gaussians=n_gaussians,
        n_pts_per_gaussian=n_pts_per_gaussian,
        std=std,
        batch_size=bs,
        som_sigma=som_sigma,
        net_2d=net_2d,
        add_fov_hor=add_fov_hor,
        add_fov_ver=add_fov_ver,
        sphere_W=sphere_w, sphere_H=sphere_h,
        eval_depth=eval_depth,
        use_color=use_color,
        use_reprojection=use_reprojection
    )

    if enable_log:
        logger = TensorBoardLogger(save_dir=logdir, name=exp_name, version="")
        lr_monitor = LearningRateMonitor(logging_interval="step")
        checkpoint_callbacks = [
            ModelCheckpoint(
                save_last=True,
                monitor="valdepth/abs_rel",
                save_top_k=1,
                mode="min",
                filename="{epoch:03d}-{valdepth/abs_rel:.4f}",
            ),
            lr_monitor,
        ]
    else:
        logger = False
        checkpoint_callbacks = False

    model_path = os.path.join(logdir, exp_name, "checkpoints/last.ckpt")
    if os.path.isfile(model_path):
        # Continue training from last.ckpt
        trainer = Trainer(
            # detect_anomaly=True,
            callbacks=checkpoint_callbacks,
            resume_from_checkpoint=model_path,
            sync_batchnorm=True,
            deterministic=False,
            max_epochs=max_epochs,
            gpus=n_gpus,
            logger=logger,
            check_val_every_n_epoch=1,
            log_every_n_steps=10,
            # gradient_clip_val=1.0,
            flush_logs_every_n_steps=100,
            accelerator='ddp',
        )
    else:
        # Train from scratch
        trainer = Trainer(
            # detect_anomaly=True,
            callbacks=checkpoint_callbacks,
            sync_batchnorm=True,
            deterministic=False,
            max_epochs=max_epochs,
            gpus=n_gpus,
            logger=logger,
            check_val_every_n_epoch=1,
            log_every_n_steps=10,
            flush_logs_every_n_steps=100,
            accelerator='ddp',
        )

    trainer.fit(model, data_module)
    # trainer.validate(model, data_module)



if __name__ == "__main__":
    main()
