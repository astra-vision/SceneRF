import torch

def collate_fn(batch):
    data = {}

    img_inputs = []
    img_input_sources = []
    source_distances = []
    source_frame_ids = []

    batch_img_sources = []
    batch_img_targets = []
    frame_ids = []
    sequences = []

    cam_Ks = []
    
    T_velo_2_cams = []

    batch_T_source2targets = []
    batch_T_source2infers = []

    lidar_depths = []
    loc2d_with_depths = []
    depths = []

    target_1_1s = []

    scale_3ds = batch[0]["scale_3ds"]
    for scale_3d in scale_3ds:
        data["projected_pix_{}".format(scale_3d)] = []
        data["fov_mask_{}".format(scale_3d)] = []

    for idx, input_dict in enumerate(batch):
        lidar_depths.append(input_dict["lidar_depths"])
        loc2d_with_depths.append(input_dict["loc2d_with_depths"])
        depths.append(input_dict['depths'])
        
        cam_Ks.append(torch.from_numpy(input_dict["cam_K"]).float())
        
        batch_T_source2targets.append(input_dict["T_source2targets"])
        batch_T_source2infers.append(input_dict["T_source2infers"])
        batch_img_sources.append(input_dict['img_sources'])
        batch_img_targets.append(input_dict['img_targets'])


        
        T_velo_2_cams.append(torch.from_numpy(input_dict["T_velo_2_cam"]).float())

        for key in data:
            data[key].append(torch.from_numpy(input_dict[key]))
        
        img_inputs.append(input_dict["img_input"])
        img_input_sources.append(input_dict["img_input_sources"])
        source_distances.append(input_dict['source_distances'])
        source_frame_ids.append(input_dict['source_frame_ids'])
        
        
        frame_ids.append(input_dict["frame_id"])
        sequences.append(input_dict["sequence"])

 
        if "target_1_1" in input_dict:
            target_1_1s.append(torch.from_numpy(input_dict["target_1_1"]))

    ret_data = {
        "T_source2targets": batch_T_source2targets, 
        "T_source2infers": batch_T_source2infers,

        "loc2d_with_depths": loc2d_with_depths,
        "lidar_depths": lidar_depths,
        "depths": depths,

        "frame_id": frame_ids,
        "sequence": sequences,
        "cam_K": torch.stack(cam_Ks),

        "T_velo_2_cam": torch.stack(T_velo_2_cams),
        

        "img_inputs": torch.stack(img_inputs),
        "img_input_sources": img_input_sources,
        "source_distances": source_distances,
        "source_frame_ids": source_frame_ids,
        
        "img_sources": batch_img_sources,
        "img_targets": batch_img_targets,
        
    }
    if len(target_1_1s) > 0:
        ret_data["target_1_1"] = torch.stack(target_1_1s)
    
    for key in data:
        ret_data[key] = data[key]
    return ret_data
