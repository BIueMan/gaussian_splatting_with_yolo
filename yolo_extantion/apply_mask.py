import numpy as np
import cv2
import os

# Load the .npy file containing the masks
mask_file_path = 'output/mask_output.npy'
masks = np.load(mask_file_path)

# Directory paths
image_dir = 'data/to_apply_mask'
output_dir = 'output/apply_mask'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Iterate through the images and corresponding masks
for i in range(16):
    # Read the image
    image_number = f"{i+1:03d}.png"
    image_path = os.path.join(image_dir, image_number)
    image = cv2.imread(image_path)

    print(image_number)
    if image is None:
        print(f"Image {image_number} could not be read.")
        continue

    # Get the corresponding mask
    mask = masks[i]

    # Resize the mask to the shape of the image
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # Threshold the mask to binary (0 or 1)
    mask_boolean = mask_resized > 0
    mask_float = mask_boolean.astype(np.float32)

    # Apply the mask to the image
    masked_image = image * mask_float[:, :, np.newaxis]

    # Save the masked image
    output_path = os.path.join(output_dir, f"{image_number}")
    cv2.imwrite(output_path, masked_image)

    print(f"Processed and saved: {output_path}")
