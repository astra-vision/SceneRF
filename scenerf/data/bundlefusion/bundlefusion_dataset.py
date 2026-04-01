import torch
import os
import glob
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
import imageio


class BundlefusionDataset(Dataset):
    def __init__(
        self,
        split,
        dataset,
        root,
        n_sources=1,
        frame_interval=4,
        n_frames=16,
        infer_frame_interval=2,
        color_jitter=None,
        select_scans=None,
        tum_rgbd=False,
    ):
        self.root = root

        print(dataset)
        # Select a split based on training dataset being either bf or tum_rgbd
        if dataset == "bf":
            splits = {
                "train": ["apt0", "apt1", "apt2", "office0", "office1", "office2", "office3"],
                "val": ["copyroom"],
                "all": ["apt0", "apt1", "apt2", "office0", "office1", "office2", "office3", "copyroom"]
            }

        elif dataset == "tum_rgbd":
            splits = {
                "train": [
                    "rgbd_dataset_freiburg1_360",
                    "rgbd_dataset_freiburg1_desk",
                    "rgbd_dataset_freiburg1_floor",
                    "rgbd_dataset_freiburg1_room",
                    "rgbd_dataset_freiburg1_xyz",
                    "rgbd_dataset_freiburg2_360_hemisphere",
                    "rgbd_dataset_freiburg2_desk",
                    "rgbd_dataset_freiburg2_large_no_loop",
                    "rgbd_dataset_freiburg2_pioneer_360",
                    "rgbd_dataset_freiburg2_xyz",
                    "rgbd_dataset_freiburg3_structure_texture_far"
                ],
                "val": [
                    "rgbd_dataset_freiburg3_long_office_household"
                ],
                "all": [
                    "rgbd_dataset_freiburg1_360",
                    "rgbd_dataset_freiburg1_desk",
                    "rgbd_dataset_freiburg1_floor",
                    "rgbd_dataset_freiburg1_room",
                    "rgbd_dataset_freiburg1_xyz",
                    "rgbd_dataset_freiburg2_360_hemisphere",
                    "rgbd_dataset_freiburg2_desk",
                    "rgbd_dataset_freiburg2_large_no_loop",
                    "rgbd_dataset_freiburg2_pioneer_360",
                    "rgbd_dataset_freiburg2_xyz",
                    "rgbd_dataset_freiburg3_structure_texture_far",
                    "rgbd_dataset_freiburg3_long_office_household"
                ]
            }

        self.sequences = splits[split]
        self.n_sources = n_sources
        self.frame_interval = frame_interval
        self.n_frames = n_frames
        self.infer_frame_interval = infer_frame_interval
        print("n_frames: ", self.n_frames)
        print("frame_interval: ", self.frame_interval)
        self.color_jitter = color_jitter

        self.img_W = 640
        self.img_H = 480
        # ids = []
        self.error_frames = []
        error_frames_path = os.path.join(os.path.dirname(__file__), "error_frames.txt")
        with open(error_frames_path, "r") as file:
            for line in file:
                self.error_frames.append(line.strip())

        self.scans = []
        for sequence in self.sequences:
            cam_K_color, cam_K_depth = self.read_camera_params(
                os.path.join(self.root, sequence, "info.txt")
            )
            glob_path = os.path.join(
                self.root, sequence, "*.color.jpg"
            )
            rgb_paths = glob.glob(glob_path)
            for rgb_path in rgb_paths:
                filename = os.path.basename(rgb_path)
                frame_id = float(os.path.splitext(filename)[0][6:12])
                frame_id_with_sequence = sequence + "_" + "{:06d}".format(int(frame_id))
                if frame_id_with_sequence in self.error_frames:
                    continue
                if (frame_id % self.infer_frame_interval) != 0:
                    continue
                if frame_id < self.n_frames // 2 * self.frame_interval:
                    continue
                if frame_id > (len(rgb_paths) - 1 - self.n_frames // 2 * self.frame_interval):
                    continue
                rel_frame_ids = []
                for i in range(-self.n_frames // 2, self.n_frames // 2 + 1):
                    rel_frame_id = "{:06d}".format(int(frame_id) + i * self.frame_interval)
                    rel_frame_ids.append(rel_frame_id)

                
                if select_scans is not None and rel_frame_ids[self.n_frames // 2] not in select_scans:
                    continue
                self.scans.append({
                    "sequence": sequence,
                    # "frame_id": "{:06d}".format(int(frame_id)),
                    "rel_frame_ids": rel_frame_ids,
                    "cam_K_color": cam_K_color,
                    "cam_K_depth": cam_K_depth
                })
                
        self.color_jitter = (
            transforms.ColorJitter(*color_jitter) if color_jitter else None
        )            

        self.normalize_rgb = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.to_tensor_normalized = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.to_tensor = transforms.Compose([
            transforms.ToTensor()
        ])

        print("n_scans", len(self.scans))


    def __getitem__(self, index):
        scan = self.scans[index]        
        sequence = scan['sequence']
        rel_frame_ids = scan['rel_frame_ids']
        infer_id = self.n_frames // 2
        frame_id = rel_frame_ids[infer_id]
        infer_pose_path = os.path.join(self.root, sequence, "frame-{}.pose.txt".format(rel_frame_ids[infer_id]))
        
        
        img_sources = []
        img_targets = []

        T_source2infers = []
        T_source2targets = []
        source_frame_ids = []
        source_depths = []

        img_input_path = os.path.join(self.root, sequence, "frame-{}.color.jpg".format(frame_id))
        img_input = self.to_tensor_normalized(self.read_rgb(img_input_path, aug=True))
        img_input_original = self.to_tensor(self.read_rgb(img_input_path, aug=False))

        infer_depth_path = os.path.join(self.root, sequence, "frame-{}.depth.png".format(frame_id))
        infer_depth = self._read_depth(infer_depth_path)

        
        idx = np.arange(self.n_frames + 1)
        idx = np.delete(idx, infer_id)
        n_sources = min(len(idx), self.n_sources)
        for d_id in range(n_sources):
            if self.n_sources < len(rel_frame_ids):
                source_id = np.random.choice(idx, 1)[0]
            else:
                source_id = idx[d_id]

            rel_frame_id = rel_frame_ids[source_id]
            source_frame_ids.append(rel_frame_id)
            target_id = source_id - 1
            
            img_source_path = os.path.join(self.root, sequence, "frame-{}.color.jpg".format(rel_frame_ids[source_id]))
            img_target_path = os.path.join(self.root, sequence, "frame-{}.color.jpg".format(rel_frame_ids[target_id]))
            img_source = self.to_tensor(self.read_rgb(img_source_path))
            img_target = self.to_tensor(self.read_rgb(img_target_path))
            img_sources.append(img_source)
            img_targets.append(img_target)


            source_pose_path = os.path.join(self.root, sequence, "frame-{}.pose.txt".format(rel_frame_ids[source_id]))
            target_pose_path = os.path.join(self.root, sequence, "frame-{}.pose.txt".format(rel_frame_ids[target_id]))

            infer_pose = self.read_pose(infer_pose_path)      
            source_pose = self.read_pose(source_pose_path)
            target_pose = self.read_pose(target_pose_path)
            

            T_source2infer = np.linalg.inv(infer_pose) @ source_pose
            T_source2infers.append(torch.from_numpy(T_source2infer).float())
            T_source2target = np.linalg.inv(target_pose) @ source_pose
            T_source2targets.append(torch.from_numpy(T_source2target).float())
            
            source_depth_path = os.path.join(self.root, sequence, "frame-{}.depth.png".format(rel_frame_ids[source_id]))
            source_depth = self._read_depth(source_depth_path)
            source_depths.append(source_depth)
          
        data = {
            "sequence": sequence,
            "infer_depth": infer_depth,
            "img_input": img_input,
            "img_input_original": img_input_original,
            "source_frame_ids": source_frame_ids,

            "img_sources": img_sources,
            "img_targets": img_targets,


            "source_depths": source_depths,
            "T_source2infers": T_source2infers,
            "T_source2targets": T_source2targets,

            "frame_id": frame_id,            

            "cam_K_color": scan['cam_K_color'],
            "cam_K_depth": scan['cam_K_depth']
        }
        return data

    def __len__(self):
        return len(self.scans)


    @staticmethod
    def read_camera_params(path):
        """
        Modify from https://github.com/utiasSTARS/pykitti/blob/d3e1bb81676e831886726cc5ed79ce1f049aef2c/pykitti/utils.py#L68
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
        with open(path, "r") as f:
            for line in f.readlines():
                if line == "\n":
                    break
                key, value = line.split("=")
                key = key.strip()
                value = value.strip()
                if key == "m_calibrationColorIntrinsic":
                    cam_K_color = np.array([float(x) for x in value.split()]).reshape(4, 4)
                if key == "m_calibrationDepthIntrinsic":
                    cam_K_depth = np.array([float(x) for x in value.split()]).reshape(4, 4)

        return cam_K_color[:3, :3], cam_K_depth[:3, :3]


    def read_pose(self, path):

        # Read and parse the poses
        pose = np.identity(4)
        with open(path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                pose[i, :] = np.fromstring(line, dtype=float, sep=' ')
        
        return pose

    def read_rgb(self, path, aug=False):
        img = Image.open(path).convert("RGB")
        # print(img.size)  # (W, H)

        if aug and self.color_jitter is not None:
            img = self.color_jitter(img)

        # PIL to numpy
        img = np.array(img, dtype=np.float32, copy=False) / 255.0      

        return img


    @staticmethod
    def _read_depth(depth_filename):
        r"""Read a depth image with size H x W
        and save the depth values (in millimeters) into a 2d numpy array.
        The depth image file is assumed to be in 16-bit PNG format, depth in millimeters.
        """
        depth = imageio.imread(depth_filename)
        depth = np.asarray(depth)

        return depth
        

if __name__ == "__main__":
    root = '/gpfsdswork/dataset/bundlefusion/'

    ds = BundlefusionDataset(
        "all",
        root,
        n_sources=1000,
        frame_interval=2,
        n_frames=20
    )
    max_depth = 0
    for i in tqdm(range(len(ds))):
        ds[i]
        if i % 100 == 0:
            print(ds.error_frames)
            print("writing error frames")
            with open("error_frames.txt", "w") as file:
                for element in ds.error_frames:
                    file.write(element + "\n")
