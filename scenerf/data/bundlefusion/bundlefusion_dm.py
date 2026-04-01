from torch.utils.data.dataloader import DataLoader
from scenerf.data.bundlefusion.bundlefusion_dataset import BundlefusionDataset
from scenerf.data.bundlefusion.collate import collate_fn
import pytorch_lightning as pl
from scenerf.data.utils.torch_util import worker_init_fn


class BundlefusionDM(pl.LightningDataModule):
    def __init__(
        self,
        dataset,
        root,
        train_n_frames=16,
        val_n_frames=8,
        batch_size=1,
        num_workers=6,
        train_frame_interval=2,
        val_frame_interval=4,
        infer_frame_train_interval=4,
        infer_frame_val_interval=20,
        n_sources=1,
    ):
        super().__init__()
        self.dataset = dataset
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_n_frames = train_n_frames
        self.val_n_frames = val_n_frames
        self.n_sources = n_sources
        self.train_frame_interval = train_frame_interval
        self.val_frame_interval = val_frame_interval
        self.infer_frame_train_interval = infer_frame_train_interval
        self.infer_frame_val_interval = infer_frame_val_interval

    def setup(self, stage=None):
        self.train_ds = BundlefusionDataset(
            split="train",
            dataset=self.dataset,
            root=self.root,
            n_frames=self.train_n_frames,
            frame_interval=self.train_frame_interval,
            infer_frame_interval=self.infer_frame_train_interval,
            color_jitter=None,
            n_sources=self.n_sources
        )
        self.setup_val_ds()

    def setup_val_ds(self, select_scans=None):
        self.val_ds = BundlefusionDataset(
            split="val",
            dataset=self.dataset,
            root=self.root,
            n_frames=self.val_n_frames,
            frame_interval=self.val_frame_interval,
            infer_frame_interval=self.infer_frame_val_interval,
            color_jitter=None,
            n_sources=self.n_sources,
            select_scans=select_scans
        )
        

    def train_dataloader(self, shuffle=True):
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

    def val_dataloader(self, shuffle=False):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            shuffle=shuffle,
            pin_memory=True,
            collate_fn=collate_fn,
        )
