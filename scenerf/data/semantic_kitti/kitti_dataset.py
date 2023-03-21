import glob
import os
import pickle
import time

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from scenerf.data.utils.helpers import dump_xyz, vox2pix, read_calib, compute_transformation, read_poses, read_rgb
from scenerf.data.semantic_kitti.params import val_error_frames
import scenerf.data.semantic_kitti.io_data as SemanticKittiIO


class KittiDataset(Dataset):
    def __init__(
            self,
            split,
            root, preprocess_root,
            frames_interval=0.4,
            sequence_distance=10,
            n_sources=1,
            eval_depth=80,
            sequences=None,
            selected_frames=None, 
            n_rays=1200,
    ):
        super().__init__()
        self.root = root
        self.preprocess_root = preprocess_root
        self.depth_preprocess_root = os.path.join(preprocess_root, "depth")
        self.transform_preprocess_root = os.path.join(preprocess_root, "transform")
        self.n_classes = 20
        self.n_sources = n_sources
        self.eval_depth = eval_depth
        self.n_rays = n_rays

        splits = {
            "train": ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],                   
            "val": ["08"],
            "test": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"],
        }
        self.split = split
        if sequences is not None:
            self.sequences = sequences
        else:
            self.sequences = splits[split]
        self.output_scale = 1
        self.scene_size = (51.2, 51.2, 6.4)
        self.vox_origin = np.array([0, -25.6, -2])
        self.frames_interval = frames_interval
        self.sequence_distance = sequence_distance



        self.voxel_size = 0.2  # 0.2m
        self.img_W = 1220
        self.img_H = 370

      
        start_time = time.time()
        self.scans = []

        for sequence in self.sequences:
            pose_path = os.path.join(self.root, "dataset", "poses", sequence + ".txt")
            gt_global_poses = read_poses(pose_path)

            calib = read_calib(
                os.path.join(self.root, "dataset", "sequences", sequence, "calib.txt")
            )
            P = calib["P2"]

            T_cam0_2_cam2 = calib['T_cam0_2_cam2']
            T_cam2_2_cam0 = np.linalg.inv(T_cam0_2_cam2)
            T_velo_2_cam = T_cam0_2_cam2 @ calib["Tr"]
            
            if split == "val":
                glob_path = os.path.join(
                    self.root, "dataset", "sequences", sequence, "voxels", "*.bin"
                )

            else:
                glob_path = os.path.join(
                    self.root, "dataset", "sequences", sequence, "image_2", "*.png"
                )

            
            seq_img_paths = glob.glob(glob_path)

            max_length = 0
            min_length = 50
            for seq_img_path in seq_img_paths:
                filename = os.path.basename(seq_img_path)
                frame_id = os.path.splitext(filename)[0]
                if split == "val" and float(frame_id) % 5 != 0:
                    continue
                            
                rel_frame_ids = []

                img_paths = []
                seg2d_paths = []
                lidar_paths = []
                poses = []
                distances = []


                distance = 0
                cnt = -1
            
                while True:
                    cnt += 1
                    rel_frame_id = "{:06d}".format(int(frame_id) + cnt)

                    img_path = os.path.join(
                        self.root, "dataset", "sequences", sequence, "image_2", rel_frame_id + ".png"
                    )

                    should_add = os.path.exists(img_path)
                    if not should_add:
                        break
   
                    current_pose = gt_global_poses[int(rel_frame_id)]
                    if len(poses) > 0:
                        prev_pose = poses[-1]
                        prev_xyz = dump_xyz(prev_pose)
                        current_xyz = dump_xyz(current_pose)
                        rel_distance = np.sqrt(
                            (prev_xyz[0] - current_xyz[0]) ** 2 + (prev_xyz[2] - current_xyz[2]) ** 2)
                        distance += rel_distance
                        if rel_distance < frames_interval:
                            continue
                        if distance > self.sequence_distance:
                            break

                    rel_frame_ids.append(rel_frame_id)
                    img_paths.append(img_path)
                    poses.append(current_pose)
                    distances.append(distance)

                    # We use lidar for evaluation only
                    lidar_path = os.path.join(self.root, "dataset", "sequences", sequence, "velodyne",
                                              rel_frame_id + ".bin")
                    lidar_paths.append(lidar_path)


                if len(poses) == 1:
                    should_add = False
                # ignore error frame
                if split == "val" and frame_id in val_error_frames:
                    should_add = False

                is_included = False
                if selected_frames is not None:
                    is_included = frame_id in selected_frames
                else:
                    is_included = should_add

                if is_included:
                    
                    if len(poses) > max_length:
                        max_length = len(poses)
                    if len(poses) < min_length:
                        min_length = len(poses)

                    self.scans.append(
                        {
                            "frame_id": frame_id,
                            "sequence": sequence,

                            "img_paths": img_paths,
                            "seg2d_paths": seg2d_paths,
                            "lidar_paths": lidar_paths,

                            "T_velo_2_cam": T_velo_2_cam,
                            "P": P,
                            "T_cam0_2_cam2": T_cam0_2_cam2,
                            "T_cam2_2_cam0": T_cam2_2_cam0,

                            "poses": np.stack(poses, axis=0),
                            "distances": distances,
                            "rel_frame_ids": rel_frame_ids
                        }
                    )
            print(sequence, min_length, max_length)

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
        print("Preprocess time: --- %s seconds ---" % (time.time() - start_time))


    def get_depth_from_lidar(self, lidar_path, P, T_velo_2_cam, image_size):
        scan = np.fromfile(lidar_path, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        points = scan[:, :3]

        keep_idx = points[:, 0] > 0  # only keep point in front of the vehicle
        points_hcoords = np.concatenate([points[keep_idx], np.ones([keep_idx.sum(), 1], dtype=np.float32)], axis=1)

        pts_cam = (T_velo_2_cam @ points_hcoords.T).T
        # print(pts_cam[:, 2].min(), pts_cam[:, 2].max())
        mask = (pts_cam[:, 2] <= self.eval_depth) & (pts_cam[:, 2] > 0)  # get points with depth < max_sample_depth
        pts_cam = pts_cam[mask, :3]

        img_points = (P[0:3, 0:3] @ pts_cam.T).T

        img_points = img_points[:, :2] / np.expand_dims(img_points[:, 2], axis=1)  # scale 2D points
        img_points = np.round(img_points).astype(int)
        keep_idx_img_pts = (img_points[:, 0] > 0) & \
                           (img_points[:, 1] > 0) & \
                           (img_points[:, 0] < image_size[0]) & \
                           (img_points[:, 1] < image_size[1])

 
        img_points = img_points[keep_idx_img_pts, :]

        pts_cam = pts_cam[keep_idx_img_pts, :]

        depths = pts_cam[:, 2]

        return img_points, depths, pts_cam

    def __getitem__(self, index):
        scan = self.scans[index]
        frame_id = scan['frame_id']
        sequence = scan['sequence']
        lidar_paths = scan['lidar_paths']
        rel_frame_ids = scan['rel_frame_ids']
        distances = scan['distances']
        infer_id = 0
        

        P = scan["P"]
        T_velo_2_cam = scan["T_velo_2_cam"]

        img_paths = scan["img_paths"]

        img_sources = []
        img_input_sources = []
        img_targets = []
        lidar_depths = []
        depths = []
        loc2d_with_depths = []
        T_source2infers = []
        T_source2targets = []
        source_distances = []
        source_frame_ids = []

        
        n_sources = min(len(distances) - 1, self.n_sources)
        
        for d_id in range(n_sources):
            if self.n_sources < len(distances):    
                source_id = np.random.randint(1, len(distances))
                source_distance = distances[source_id]  
            else:
                source_id = d_id + 1
                source_distance = distances[source_id]
        
            source_distances.append(source_distance)

            rel_frame_id = rel_frame_ids[source_id]
            source_frame_ids.append(rel_frame_id)

            target_id = source_id - 1

            img_input_source = self.to_tensor_normalized(read_rgb(img_paths[source_id]))
            img_input_sources.append(img_input_source)

       
            img_source = self.to_tensor(read_rgb(img_paths[source_id]))
            img_target = self.to_tensor(read_rgb(img_paths[target_id]))


            lidar_path = lidar_paths[source_id]
            loc2d_with_depth, lidar_depth, _ = self.get_depth_from_lidar(lidar_path, P, T_velo_2_cam,
                                                                         (self.img_W, self.img_H))

            if self.n_rays  < lidar_depth.shape[0]:
                idx = np.random.choice(lidar_depth.shape[0], size=self.n_rays, replace=False)
                loc2d_with_depth = loc2d_with_depth[idx, :]
                lidar_depth = lidar_depth[idx]

            img_sources.append(img_source)
            img_targets.append(img_target)
            lidar_depths.append(torch.from_numpy(lidar_depth))
            loc2d_with_depths.append(torch.from_numpy(loc2d_with_depth))

            # Get transformation from source to target coord
            transform_dir = os.path.join(self.transform_preprocess_root,
                                         "{}_{}_all".format(sequence, self.frames_interval))
            os.makedirs(transform_dir, exist_ok=True)

            transform_path = os.path.join(transform_dir, "{}.pkl".format(frame_id))

            
            if os.path.exists(transform_path):
                try:
                    with open(transform_path, "rb") as input_file:
                        transform_data = pickle.load(input_file)
                except EOFError:
                    transform_data = {}
            else:
                transform_data = {}

            if '{}'.format(source_id) in transform_data:
                T_out_dict = transform_data['{}'.format(source_id)]
            else:
                poses = scan["poses"]
                pose_source = poses[source_id]
                pose_infer = poses[infer_id]
                pose_target = poses[target_id]
                lidar_path_source = lidar_paths[source_id]
                lidar_path_target = lidar_paths[target_id]
                lidar_path_infer = lidar_paths[infer_id]

                T_out_dict = compute_transformation(
                    lidar_path_source, lidar_path_infer, lidar_path_target,
                    pose_source, pose_infer, pose_target,
                    T_velo_2_cam, scan['T_cam0_2_cam2'])

                transform_data['{}'.format(source_id)] = T_out_dict
                with open(transform_path, "wb") as input_file:
                    pickle.dump(transform_data, input_file)
                    print("{}: saved {} to {}".format(frame_id, source_id, transform_path))

            T_source2infer = T_out_dict['T_source2infer']
            T_source2target = T_out_dict['T_source2target']
            T_source2infers.append(torch.from_numpy(T_source2infer).float())
            T_source2targets.append(torch.from_numpy(T_source2target).float())
           


        data = {

            "img_input_sources": img_input_sources,
            "source_distances": source_distances,
            "source_frame_ids": source_frame_ids,

            "img_sources": img_sources,
            "img_targets": img_targets,

            "lidar_depths": lidar_depths,
            "depths": depths,
            "loc2d_with_depths": loc2d_with_depths,
            "T_source2infers": T_source2infers,
            "T_source2targets": T_source2targets,

            "frame_id": frame_id,
            "sequence": sequence,

            "P": P,
            "T_velo_2_cam": T_velo_2_cam,
            "T_cam2_2_cam0": scan['T_cam2_2_cam0'],
            "T_cam0_2_cam2": scan['T_cam0_2_cam2'],

        }
        
        scale_3ds = [self.output_scale]
        data["scale_3ds"] = scale_3ds
        cam_K = P[0:3, 0:3]
        data["cam_K"] = cam_K
        for scale_3d in scale_3ds:
            # compute the 3D-2D mapping
           
            projected_pix, fov_mask, sensor_distance = vox2pix(
                T_velo_2_cam,
                cam_K,
                self.vox_origin,
                self.voxel_size * scale_3d,
                self.img_W,
                self.img_H,
                self.scene_size,
            )
           
            data["projected_pix_{}".format(scale_3d)] = projected_pix
            data["sensor_distance_{}".format(scale_3d)] = sensor_distance
            data["fov_mask_{}".format(scale_3d)] = fov_mask
        
        img_input = read_rgb(img_paths[infer_id])

        img_input = self.to_tensor_normalized(img_input)
        data["img_input"] = img_input
        

        label_path = os.path.join(
            self.root, "dataset", "sequences", sequence, "voxels", "{}.label".format(frame_id)
        )
        invalid_path = os.path.join(
            self.root, "dataset", "sequences", sequence, "voxels", "{}.invalid".format(frame_id)
        )
        data['target_1_1'] = self.read_semKITTI_label(label_path, invalid_path)
        
        
        return data


    @staticmethod
    def read_semKITTI_label(label_path, invalid_path):
        remap_lut = SemanticKittiIO.get_remap_lut("./scenerf/data/semantic_kitti/semantic-kitti.yaml")
        LABEL = SemanticKittiIO._read_label_SemKITTI(label_path)
        INVALID = SemanticKittiIO._read_invalid_SemKITTI(invalid_path)
        LABEL = remap_lut[LABEL.astype(np.uint16)].astype(
            np.float32
        )  # Remap 20 classes semanticKITTI SSC
       
        LABEL[
            np.isclose(INVALID, 1)
        ] = 255  # Setting to unknown all voxels marked on invalid mask...
        
        LABEL = LABEL.reshape(256,256, 32)
        return LABEL

    def __len__(self):
        return len(self.scans)

