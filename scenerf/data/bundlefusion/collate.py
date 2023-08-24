import torch
import numpy as np
import pdb


def collate_fn(batch):

    img_inputs = []
    img_input_originals = []

    source_distances = []
    source_frame_ids = []
    source_depths = []

    batch_img_sources = []
    batch_img_targets = []

    frame_ids = []

    cam_K_colors = []
    cam_K_depths = []
    
    batch_T_source2targets = []
    batch_T_source2infers = []

    sequences = []
    infer_depths = []


    for idx, input_dict in enumerate(batch):
        sequences.append(input_dict["sequence"])

        infer_depths.append(torch.from_numpy(input_dict["infer_depth"]).float())

        cam_K_colors.append(torch.from_numpy(input_dict["cam_K_color"]).float())
        cam_K_depths.append(torch.from_numpy(input_dict["cam_K_depth"]).float())
        
        batch_T_source2targets.append(input_dict["T_source2targets"])
        batch_T_source2infers.append(input_dict["T_source2infers"])

        batch_img_sources.append(input_dict['img_sources'])
        batch_img_targets.append(input_dict['img_targets'])

        source_depths.append(input_dict['source_depths'])

        img_inputs.append(input_dict["img_input"])
        img_input_originals.append(input_dict["img_input_original"])
        source_frame_ids.append(input_dict['source_frame_ids'])
        
        frame_ids.append(input_dict["frame_id"])
  

    ret_data = {
        "sequence": sequences,
        "infer_depths": torch.stack(infer_depths),

        "T_source2targets": batch_T_source2targets, 
        "T_source2infers": batch_T_source2infers,

        "frame_id": frame_ids,
      
        "cam_K_color": torch.stack(cam_K_colors),
        "cam_K_depth": torch.stack(cam_K_depths),
        
        "img_inputs": torch.stack(img_inputs),
        "img_input_originals": torch.stack(img_input_originals),

        "source_distances": source_distances,
        "source_frame_ids": source_frame_ids,
        "source_depths": source_depths,
        
        "img_sources": batch_img_sources,
        "img_targets": batch_img_targets,
        
    }
   
    
    return ret_data
