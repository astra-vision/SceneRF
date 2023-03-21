import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader

from scenerf.data.semantic_kitti.collate import collate_fn
from scenerf.data.semantic_kitti.kitti_dataset import KittiDataset
from scenerf.data.utils.torch_util import worker_init_fn


class KittiDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root,
        preprocess_root,
        batch_size=4,
        num_workers=4,
        source_id=0,
        sequence_distance=10,
        eval_depth=80,
        frames_interval=0.4,
        n_sources=1,
        n_rays=1200,
        selected_frames=None
    ):
        super().__init__()
        self.root = root
        self.preprocess_root = preprocess_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.source_id = source_id
        self.frames_interval = frames_interval
        self.sequence_distance = sequence_distance
        self.eval_depth = eval_depth
        self.n_rays = n_rays
        self.selected_frames = selected_frames
        self.n_sources = n_sources

    def setup_train_ds(self):
        self.train_ds = KittiDataset(
            split="train",
            root=self.root,
            preprocess_root=self.preprocess_root,
            sequence_distance=self.sequence_distance,
            n_sources=self.n_sources,
            frames_interval=self.frames_interval,
            selected_frames=self.selected_frames,
            eval_depth=self.eval_depth,
            n_rays=self.n_rays
        )

    def setup_val_ds(self):
        self.val_ds = KittiDataset(
            split="val",
            root=self.root,
            n_sources=self.n_sources,
            preprocess_root=self.preprocess_root,
            sequence_distance=self.sequence_distance,
            eval_depth=self.eval_depth,
            frames_interval=self.frames_interval,
            selected_frames=self.selected_frames,
            n_rays=self.n_rays
        )

    def setup(self, stage=None):
        self.setup_train_ds()
        self.setup_val_ds()

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
        )

