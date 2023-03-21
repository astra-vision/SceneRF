import os
import click
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from scenerf.data.semantic_kitti.kitti_dm import KittiDataModule
from scenerf.models.scenerf import scenerf

seed_everything(42)


@click.command()
@click.option('--dataset', default="kitti", help='experiment prefix')
@click.option('--logdir', default="", help='log directory')
@click.option('--root', default="", help='path to dataset folder')
@click.option('--preprocess_root', default="", help='path to preprocess folder')
@click.option('--bs', default=1, help='Batch size')
@click.option('--lr', default=1e-5, help='learning rate')
@click.option('--wd', default=0, help='weight decay')
@click.option('--n_gpus', default=1, help='number of GPUs')
@click.option('--n_workers_per_gpu', default=4, help='number of workers per GPU')
@click.option('--enable_log', default=False, help='enable log')
@click.option('--exp_prefix', default="exp", help='experiment prefix')

@click.option('--n_rays', default=1200, help='Total number of rays')
@click.option('--frames_interval', default=0.4, help='Interval between supervision frames')

@click.option('--max_sample_depth', default=100, help='maximum sample depth')
@click.option('--eval_depth', default=80, help='cap depth at 80 for evaluation')

@click.option('--n_pts_per_gaussian', default=8, help='#points sampled for each gaussian')
@click.option('--n_gaussians', default=4, help='#gaussians')
@click.option('--n_pts_uni', default=32, help='#points sampled uniformly')
@click.option('--std', default=2.0, help='initial std of each gaussian')
@click.option('--add_fov_hor', default=20, help='Amount of angle in degree added to left and right of the horizontal FOV')
@click.option('--add_fov_ver', default=8, help='Amount of angle in degree added to top and bottom of the vertical FOV')
# ideally sphere_h and sphere_w should be img_H * 1.5, img_W * 1.5 (Because we increase the FOV by 1.5). 
# However, we empirically found that any sphere_h >= img_H and any sphere_w >= img_W have almost similar performance. 
@click.option('--sphere_h', default=452, help='The height of the discretized spherical grid') 
@click.option('--sphere_w', default=1500, help='The width of the discretized spherical grid') 
@click.option('--sequence_distance', default=10, help='Distance between the input and the last frames in the sequence')
@click.option('--som_sigma', default=2.0, help='')
@click.option('--max_epochs', default=20, help='')
@click.option('--use_color', default=True, help='Use color loss')
@click.option('--use_reprojection', default=True, help='Use reprojection loss')
def main(
        dataset, root, preprocess_root,
        bs, n_gpus, n_workers_per_gpu,
        exp_prefix,
        logdir, enable_log,
        lr, wd,
        n_rays, 
        frames_interval, 
        sequence_distance,
        max_sample_depth, eval_depth,
        n_pts_uni,
        n_pts_per_gaussian, n_gaussians, std, som_sigma,
        add_fov_hor, add_fov_ver,
        use_color, use_reprojection,
        sphere_w, sphere_h, max_epochs):
    exp_name = exp_prefix
    exp_name += "_lr{}_{}rays".format(lr, n_rays)
    exp_name += "_nGaus{}_nPtsPerGaus{}_std{}_SOMSigma{}".format(n_gaussians, n_pts_per_gaussian, std, som_sigma)
    exp_name += "_sphere{}x{}_addfov{}x{}".format(sphere_w, sphere_h, add_fov_hor, add_fov_ver)
    
    if not use_reprojection:
        exp_name += "NoReproj"
    if not use_color:
        exp_name += "NoColor"


       
    data_module = KittiDataModule(
        root=root,
        preprocess_root=preprocess_root,
        frames_interval=frames_interval,
        batch_size=int(bs / n_gpus),
        sequence_distance=sequence_distance,
        num_workers=int(n_workers_per_gpu),
        n_rays=n_rays,
        eval_depth=eval_depth
    )


    model = scenerf(
        dataset=dataset,
        lr=lr,
        n_pts_uni=n_pts_uni,
        weight_decay=wd,
        n_rays=n_rays,
        max_sample_depth=max_sample_depth,
        n_gaussians=n_gaussians,
        n_pts_per_gaussian=n_pts_per_gaussian,
        std=std,
        batch_size=bs,
        som_sigma=som_sigma,    
        add_fov_hor=add_fov_hor,
        add_fov_ver=add_fov_ver,
        sphere_W=sphere_w, sphere_H=sphere_h,
        eval_depth=eval_depth,
        use_color=use_color,
        use_reprojection=use_reprojection,
    )

    if enable_log:
        logger = TensorBoardLogger(save_dir=logdir, name=exp_name, version="")
        lr_monitor = LearningRateMonitor(logging_interval="step")
        checkpoint_callbacks = [
            ModelCheckpoint(
                save_last=True,
                monitor="valdepth/abs_rel",
                save_top_k=1,
                mode="max",
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
            callbacks=checkpoint_callbacks,
            resume_from_checkpoint=model_path,
            sync_batchnorm=True,
            deterministic=False,
            max_epochs=max_epochs,
            gpus=n_gpus,
            logger=logger,
            limit_train_batches=0.5,
            check_val_every_n_epoch=1,
            log_every_n_steps=10,
            flush_logs_every_n_steps=100,
            accelerator='ddp',
        )
        
    else:
        # Train from scratch
        trainer = Trainer(
            callbacks=checkpoint_callbacks,
            sync_batchnorm=True,
            deterministic=False,
            max_epochs=max_epochs,
            limit_train_batches=0.5,
            gpus=n_gpus,
            logger=logger,
            check_val_every_n_epoch=1,
            log_every_n_steps=10,
            flush_logs_every_n_steps=100,
            accelerator='ddp',
        )

    trainer.fit(model, data_module)



if __name__ == "__main__":
    main()
