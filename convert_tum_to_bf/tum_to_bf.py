import os
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
import argparse


def combine_and_rename_files(
    tum_folder: str,
    output_folder: str,
    margin: float = 0.02
) -> None:
    """
    Match and rename RGB/depth files and write them to the output folder with
    the BundleFusion naming convention. Additionally, load poses from TUM's
    groundtruth.txt, match them by timestamp within a given margin, and save
    them as 4x4 transformation matrices. Color images are re-encoded as JPG at
    maximum quality, and depth images are divided by 5 becuase for bf depth
    1mm=1 and for tum_rgbd 1mm=5.

    Parameters
    ----------
    tum_folder : str
        Path to the TUM scene folder containing 'rgb', 'depth', and
        'groundtruth.txt'.
    output_folder : str
        Path to the output folder where converted data will be stored.
    margin : float, optional
        Maximum allowed time difference (in seconds) for matching frames
        between RGB, depth, and pose data. Defaults to 0.02.

    Returns
    -------
    None
    """

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Paths to subfolders and files
    rgb_path = os.path.join(tum_folder, "rgb")
    depth_path = os.path.join(tum_folder, "depth")
    pose_file = os.path.join(tum_folder, "groundtruth.txt")

    # If any required directory or file doesn't exist, return early
    if not (os.path.isdir(rgb_path) and os.path.isdir(depth_path) and
            os.path.exists(pose_file)):
        print(f"Skipping {tum_folder} because it lacks required folders/files.")
        return

    # Get sorted lists of RGB and depth files
    rgb_files = sorted(f for f in os.listdir(rgb_path)
                       if f.lower().endswith((".png", ".jpg")))
    depth_files = sorted(f for f in os.listdir(depth_path)
                         if f.lower().endswith(".png"))

    # Extract timestamps from filenames (assuming `timestamp.ext`)
    rgb_entries = [(float(f.rsplit(".", 1)[0]), f) for f in rgb_files]
    depth_entries = [(float(f.rsplit(".", 1)[0]), f) for f in depth_files]

    # Load pose entries (timestamp tx ty tz qx qy qz qw) ignoring commented lines
    with open(pose_file, "r") as f:
        pose_lines = [
            line.strip() for line in f
            if line.strip() and not line.startswith("#")
        ]
    pose_entries = []
    for line in pose_lines:
        parts = line.split()
        ts = float(parts[0])
        data = parts[1:]  # [tx, ty, tz, qx, qy, qz, qw]
        pose_entries.append((ts, data))

    frame_counter = 0

    # Iterate over RGB frames and find matching depth and pose
    for rgb_ts, rgb_filename in rgb_entries:
        frame_id = f"frame-{frame_counter:06d}"

        # Find closest depth frame
        if not depth_entries:
            break
        closest_depth = min(depth_entries, key=lambda x: abs(rgb_ts - x[0]))
        if abs(rgb_ts - closest_depth[0]) > margin:
            continue

        # Find closest pose
        if not pose_entries:
            break
        closest_pose = min(pose_entries, key=lambda x: abs(rgb_ts - x[0]))
        if abs(rgb_ts - closest_pose[0]) > margin:
            continue

        # We have matched depth and pose; remove them from the pool
        depth_entries.remove(closest_depth)
        pose_entries.remove(closest_pose)

        # Increment the frame counter now that we have a valid match
        frame_counter += 1

        # -- Process and save color image as JPG with max quality --
        rgb_src = os.path.join(rgb_path, rgb_filename)
        rgb_dst = os.path.join(output_folder, f"{frame_id}.color.jpg")

        rgb_img = Image.open(rgb_src).convert("RGB")
        rgb_img.save(rgb_dst, "JPEG", quality=100)

        # -- Process and save depth image (divide by 5) as PNG --
        depth_src = os.path.join(depth_path, closest_depth[1])
        depth_dst = os.path.join(output_folder, f"{frame_id}.depth.png")

        depth_img = np.array(Image.open(depth_src))
        # Convert to uint16 and divide by 5 (integer division)
        depth_img = depth_img.astype(np.uint16)
        depth_img //= 5

        depth_img_pil = Image.fromarray(depth_img)
        depth_img_pil.save(depth_dst)

        # -- Process and save pose as .pose.txt --
        tx, ty, tz, qx, qy, qz, qw = map(float, closest_pose[1])
        rotation = R.from_quat([qx, qy, qz, qw]).as_matrix()

        pose_matrix = np.eye(4, dtype=np.float64)
        pose_matrix[:3, :3] = rotation
        pose_matrix[:3, 3] = [tx, ty, tz]

        pose_dst = os.path.join(output_folder, f"{frame_id}.pose.txt")
        np.savetxt(pose_dst, pose_matrix, fmt="%.6f")


def generate_info_txt(output_folder: str, folder_name: str) -> None:
    """
    Generate an 'info.txt' file suitable for BundleFusion. This file includes
    camera intrinsics and extrinsics for the color and depth sensors. The
    intrinsics are selected based on the TUM dataset prefix.

    Parameters
    ----------
    output_folder : str
        Path to the output folder where 'info.txt' will be saved.
    folder_name : str
        Name of the scene folder. Used to determine if the dataset is
        'freiburg1', 'freiburg2', or 'freiburg3'.

    Returns
    -------
    None
    """

    # Intrinsics for different TUM prefixes
    intrinsics = {
        "freiburg1": "517.3 0 318.6 0 0 516.5 255.3 0 0 0 1 0 0 0 0 1",
        "freiburg2": "520.9 0 325.1 0 0 521.0 249.7 0 0 0 1 0 0 0 0 1",
        "freiburg3": "535.4 0 320.1 0 0 539.2 247.6 0 0 0 1 0 0 0 0 1"
    }

    # Default values
    color_intrinsic = "525.0 0 319.5 0 0 525.0 239.5 0 0 0 1 0 0 0 0 1"
    depth_intrinsic = "525.0 0 319.5 0 0 525.0 239.5 0 0 0 1 0 0 0 0 1"

    folder_name_lower = folder_name.lower()
    if "freiburg1" in folder_name_lower:
        color_intrinsic = intrinsics["freiburg1"]
        depth_intrinsic = intrinsics["freiburg1"]
    elif "freiburg2" in folder_name_lower:
        color_intrinsic = intrinsics["freiburg2"]
        depth_intrinsic = intrinsics["freiburg2"]
    elif "freiburg3" in folder_name_lower:
        color_intrinsic = intrinsics["freiburg3"]
        depth_intrinsic = intrinsics["freiburg3"]

    info_path = os.path.join(output_folder, "info.txt")
    with open(info_path, "w") as f:
        f.write("m_versionNumber = 4\n")
        f.write("m_sensorName = Kinect\n")
        f.write("m_colorWidth = 640\n")
        f.write("m_colorHeight = 480\n")
        f.write("m_depthWidth = 640\n")
        f.write("m_depthHeight = 480\n")
        f.write("m_depthShift = 5000\n")
        f.write(f"m_calibrationColorIntrinsic = {color_intrinsic}\n")
        f.write("m_calibrationColorExtrinsic = "
                "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1\n")
        f.write(f"m_calibrationDepthIntrinsic = {depth_intrinsic}\n")
        f.write("m_calibrationDepthExtrinsic = "
                "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1\n")


def main(source_dir: str, dest_dir: str) -> None:
    """
    Main function that processes multiple TUM scene folders within a source
    directory and saves the converted data to a destination directory.

    For each scene in `source_dir`, this function:
    1. Calls `combine_and_rename_files` to match and convert images/poses.
    2. Calls `generate_info_txt` to create the BundleFusion 'info.txt'.

    Parameters
    ----------
    source_dir : str
        Path to the directory containing multiple TUM scenes (subdirectories).
    dest_dir : str
        Path to the directory where the converted scenes will be stored.

    Returns
    -------
    None
    """

    if not os.path.isdir(source_dir):
        print(f"Source directory '{source_dir}' is not valid.")
        return

    os.makedirs(dest_dir, exist_ok=True)

    # Process each subdirectory (scene) within the source directory
    for scene_name in sorted(os.listdir(source_dir)):
        scene_path = os.path.join(source_dir, scene_name)
        if not os.path.isdir(scene_path):
            # Skip files; only process directories
            continue

        # Create corresponding directory in destination
        output_scene_path = os.path.join(dest_dir, scene_name)
        os.makedirs(output_scene_path, exist_ok=True)

        print(f"Processing scene: {scene_name}")

        # Perform matching, renaming, and saving
        combine_and_rename_files(scene_path, output_scene_path)

        # Generate info.txt
        generate_info_txt(output_scene_path, scene_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert multiple TUM RGB-D dataset scenes to BundleFusion format."
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Path to the directory containing multiple TUM scene folders."
    )
    parser.add_argument(
        "--dest_dir",
        type=str,
        required=True,
        help="Path to the directory where converted scenes will be stored."
    )

    args = parser.parse_args()
    main(args.source_dir, args.dest_dir)

