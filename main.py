import os
import shutil
import subprocess

from yolo_extantion import *

def main(video_path, segment_idx):
    # Define paths
    yolo_dir = 'input_data/yolo_input'
    yolo_input_dir = f'{yolo_dir}/input'
    yolo_images_dir = 'output/yolo/images'
    
    # Delete yolo_input directory if it exists
    if os.path.exists(yolo_dir):
        shutil.rmtree(yolo_dir)

    # Check if the input is a video or a directory
    if video_path.endswith('.mp4'):
        create_mask(video_path, segment_idx)
        # Copy the directory
        shutil.copytree(yolo_images_dir, yolo_input_dir)
    elif os.path.isdir(video_path):
        shutil.copytree(video_path, yolo_input_dir)
    else:
        raise ValueError("The input path must be a .mp4 file or a directory.")


    # Run the conversion command
    colmap_executable = 'COLMAP-3.8-windows-cuda\\COLMAP.bat'
    subprocess.run(['python', 'convert.py', '-s', yolo_dir , '--colmap_executable', colmap_executable])

    # Apply mask
    mask_file_path = 'output/yolo/mask_output.npy'
    images_dir = os.path.join(yolo_dir, 'images')
    apply_mask(mask_file_path, images_dir)

    # Save a list of directories inside 'output'
    output_dirs = [d for d in os.listdir('output') if os.path.isdir(os.path.join('output', d))]

    # Run the training script
    subprocess.run(['python', 'train.py', '-s', yolo_dir])

    # Check for a new directory in output

    new_output_dirs = [d for d in os.listdir('output') if os.path.isdir(os.path.join('output', d))]
    new_dir = list(set(new_output_dirs) - set(output_dirs))[0]

    # Check for point_cloud.ply in the expected location
    point_cloud_path = os.path.join('output', new_dir, 'point_cloud', 'iteration_30000', 'point_cloud.ply')
    if not os.path.exists(point_cloud_path):
        raise FileNotFoundError(f"Expected file not found: {point_cloud_path}")

    gs_path = point_cloud_path
    sorted_indices = get_sort_idx_base_on_color_bright(gs_path)
    new_dir_path = os.path.join('output', new_dir, 'point_cloud', 'clean_gs')
    os.makedirs(new_dir_path, exist_ok=True)
    save_gs_path = os.path.join(new_dir_path, 'point_cloud_clean.ply')
    remove_idx_from_ply_file(gs_path, sorted_indices, 0, 4000, save_gs_path)

    print(f"Task completed. File is saved at {save_gs_path}")

# Example usage
if __name__ == '__main__':
    video_path = 'input_data/videos/plant.mp4'
    segment_idx = 1
    main(video_path, segment_idx)
