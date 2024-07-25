import numpy as np
import cv2
import os


def apply_mask(mask_file_path:str, images_dir:str):
    ##### make sure `images_dir` have only the images in it ####
    masks = np.load(mask_file_path)
    output_dir = images_dir
    
    # find max idx image
    max_image = 0
    for file_name in os.listdir(images_dir):
        max_image = max(int(file_name.split('.')[0]), max_image)

    for i in range(max_image):
        # Read the image
        image_number = f"{i+1:03d}.png"
        image_path = os.path.join(images_dir, image_number)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Image {image_number} could not be read.")
            continue

        # Resize mask to the shape of the image
        mask = masks[i]
        mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
        # make sure the Threshold of the mask are binary (0 or 1)
        mask_boolean = mask_resized > 0
        mask_float = mask_boolean.astype(np.float32)
        # Apply the mask
        masked_image = image * mask_float[:, :, np.newaxis]
        # Save the masked image
        output_path = os.path.join(output_dir, f"{image_number}")
        cv2.imwrite(output_path, masked_image)

    print(f"Processed and saved: {output_dir}")

if __name__ == "__main__":
    mask_file_path = 'output/yolo/mask_output.npy'
    images_dir = 'input_data/test/images'
    apply_mask(mask_file_path, images_dir)