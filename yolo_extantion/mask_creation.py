from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def colorize_matrix(matrix):
    # Define colors
    dark_blue = (0, 0, 139)  # Dark blue in RGB format
    yellow = (255, 255, 0)   # Yellow in RGB format
    
    # Create an empty image matrix
    height, width = matrix.shape
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Convert matrix values to colors
    for i in range(height):
        for j in range(width):
            if matrix[i, j] == 0:
                image[i, j] = dark_blue
            else:
                image[i, j] = yellow
    
    return image

def create_mask(input_video_oath:str, selected_object_idx:int, imshow_scale:float):
    # check for video size
    cap_tmp = cv2.VideoCapture(input_video_oath)
    _, frame_tnmp = cap_tmp.read()
    video_shape = np.flip(frame_tnmp.shape[:2])
    video_shape[0] = 2*video_shape[0]
    cap_tmp.release()

    # check dir are existing
    selected_object_idx -= 1
    dir_to_create = ['output', 'output/yolo', 'output/yolo/images']
    for dir in dir_to_create:
        os.makedirs(dir, exist_ok=True)
    # setup for video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30.0
    output_file = f'{dir_to_create[1]}/plant_plus_mask.mp4'
    out = cv2.VideoWriter(output_file, fourcc, fps, video_shape)
    # added text to video
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)
    thickness = 2
    line_type = cv2.LINE_AA

    # Load the YOLOv8 segmentation model
    model = YOLO("yolov8l-seg.pt")
    class_names = model.module.names if hasattr(model, 'module') else model.names

    # Open the video file
    cap = cv2.VideoCapture(input_video_oath)

    # list to save
    images_list = []
    save_mask_list = []
    box_list = []

    frame_idx = -1
    while cap.isOpened():
        # for every frame
        frame_idx +=1
        campter_frame = ((frame_idx)%int(fps) == 0)
        ret, frame = cap.read()
        if not ret:
            frame = frame_minus1
            break
        frame_minus1 = frame

        # Run YOLOv8
        results = model.track(frame, persist=True)
        frame_ = results[0].plot()
        wanted_mask = results[0].masks[selected_object_idx].data.cpu().numpy()[0]
        wanted_mask_color = colorize_matrix(wanted_mask)
        object_name_idx = int(results[0].boxes.cls[selected_object_idx].numpy())
        object_name = class_names[object_name_idx]    
        
        # save segment box
        box = results[0].boxes[selected_object_idx]
        x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
        x = int(x_min)
        y = int(y_min)
        w = int(x_max - x_min)
        h = int(y_max - y_min)
        
        # plot frame + mask
        mask_reshape = cv2.resize(wanted_mask_color, [frame_.shape[1], frame_.shape[0]])
        combined_image = cv2.hconcat([frame_, mask_reshape])
        cv2.putText(combined_image, object_name, [frame_.shape[0], 50], font, font_scale, font_color, thickness, line_type)

        cv2.imshow('frame', cv2.resize(combined_image, (np.flip(combined_image.shape[:2])/imshow_scale).astype(int)))
        out.write(np.clip(combined_image, 0, 255).astype(np.uint8))
        
        # save every sec (when campter_frame == true)
        if campter_frame:
            images_list.append(frame)
            save_mask_list.append(wanted_mask)
            box_list.append({'x': x, 'y':y, 'x_w':w, 'y_h':h})
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # add final frame
    images_list.append(frame)
    save_mask_list.append(wanted_mask)
    box_list.append({'x': x, 'y':y, 'x_w':w, 'y_h':h})
    
    # save values
    np.save(f'{dir_to_create[1]}/mask_output.npy', np.array(save_mask_list))
    np.save(f'{dir_to_create[1]}/seg_box_output.npy', np.array(box_list))
    for idx in range(len(images_list)):
        if idx +1 >= 10:
            cv2.imwrite(f'{dir_to_create[2]}/0{idx+1}.png', images_list[idx])
        else:
            cv2.imwrite(f'{dir_to_create[2]}/00{idx+1}.png', images_list[idx])

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    return 'all file save under output'

def save_small_image(seg_box_path:str, mask_path:str, images_dir_path:str):
    box_list = np.load(seg_box_path)
    save_mask_list = np.load(mask_path)
    images_list = [cv2.imread(os.path.join(images_dir_path, f)) for f in sorted(os.listdir(images_dir_path), key=lambda x: int(x.split('.')[0]))]
    # get max w,h
    max_x_w = np.max([box['x_w'] for box in box_list])
    max_y_h = np.max([box['y_h'] for box in box_list])
    image_list_small_clear = []
    image_list_small_all = []

    image_shape = images_list[0].shape[:2]

    for idx in range(len(save_mask_list)):
        imag = images_list[idx]
        mask = cv2.resize(save_mask_list[idx], (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        mask = np.stack([mask] * 3, axis=-1)
        clear_image = imag*mask
        
        x, y = box_list[idx]['x'], box_list[idx]['y']
        small_image = clear_image[y:y+max_y_h, x:x+max_x_w]
        image_list_small_clear.append(small_image)
        image_list_small_all.append(imag[y:y+max_y_h, x:x+max_x_w])
        
    for idx in range(len(image_list_small_clear)):
        if idx +1 >= 10:
            cv2.imwrite(f'output/yolo/small_clean/0{idx+1}.png', image_list_small_clear[idx])
            cv2.imwrite(f'output/yolo/small/0{idx+1}.png', image_list_small_all[idx])
        else:
            cv2.imwrite(f'output/yolo/small_clean/00{idx+1}.png', image_list_small_clear[idx])
            cv2.imwrite(f'output/yolo/small/00{idx+1}.png', image_list_small_all[idx])
            
if __name__ == "__main__":
    video_path = 'input_data/videos/car1.mp4'
    create_mask(video_path, 1, 3)