import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement
# from utils.sh_utils import SH2RGB
import tkinter as tk
from tkinter import messagebox

def SH2RGB(sh):
    C0 = 0.28209479177387814
    return sh * C0 + 0.5

def get_bounds(array_length):
    bounds = {"lower": None, "upper": None}  # Dictionary to store bounds

    def on_submit():
        try:
            lower_bound = int(lower_entry.get())
            upper_bound = int(upper_entry.get())

            if 0 <= lower_bound < upper_bound <= array_length:
                bounds["lower"] = lower_bound
                bounds["upper"] = upper_bound
                window.quit()
            else:
                messagebox.showerror("Invalid Bounds", "Please ensure 0 <= lower < upper <= array length.")
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid integers for the bounds.")

    window = tk.Tk()
    window.title("Update Bounds")

    tk.Label(window, text=f"Enter bounds within array length {array_length}, thouse element will be removed:").pack()

    tk.Label(window, text="Lower Bound:").pack()
    lower_entry = tk.Entry(window)
    lower_entry.pack()

    tk.Label(window, text="Upper Bound:").pack()
    upper_entry = tk.Entry(window)
    upper_entry.pack()

    submit_button = tk.Button(window, text="Submit", command=on_submit)
    submit_button.pack()

    window.mainloop()

    return bounds["lower"], bounds["upper"]

def get_sort_idx_base_on_color_bright(gs_path:str)->np.ndarray:
    plydata = PlyData.read(gs_path)

    lenght = np.asarray(plydata.elements[0]["x"]).shape[0]
    features_dc = np.zeros([lenght, 3, 1])
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
    rgb = SH2RGB(features_dc[..., 0])
    
    # clamp rgb and norm to 0 to 1
    rgb = np.maximum(rgb, 0)
    rgb = rgb / np.max(rgb)
    sum_color_matrix = np.sum(rgb, axis = 1)

    # Sort the distances and keep the original indices
    sorted_indices = np.argsort(sum_color_matrix)
    sorted_distances = sum_color_matrix[sorted_indices]

    # Plot the sorted distances
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_distances)
    plt.title('bright of Points')
    plt.xlabel('Point Index (Sorted)')
    plt.ylabel('Distance from Mean')
    plt.grid(True)
    plt.show()
    
    return sorted_indices

def get_sort_idx_base_on_dist_from_mean(gs_path:str)->np.ndarray:
    plydata = PlyData.read(gs_path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)

    norm_dist_matrix = np.linalg.norm(xyz, axis=1)

    # Sort the distances and keep the original indices
    sorted_indices = np.argsort(norm_dist_matrix)
    sorted_distances = norm_dist_matrix[sorted_indices]

    # Plot the sorted distances
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_distances)
    plt.title('Distances of Points from Mean Location')
    plt.xlabel('Point Index (Sorted)')
    plt.ylabel('Distance from Mean')
    plt.grid(True)
    plt.show()
    
    return sorted_indices

def remove_idx_from_ply_file(gs_path:str, sorted_indices:np.ndarray, start_indices:int, stop_indices:int, save_gs_path:str):
    # Identify points to remove
    to_remove_idx = sorted_indices[start_indices:stop_indices]
    ply_data = PlyData.read(gs_path)

    # For each property, remove the elements in to_remove_idx
    for element in ply_data.elements:
        if element.name == 'vertex':
            # Convert element data to a structured array for easier manipulation
            data = element.data
            mask = np.ones(data.shape[0], dtype=bool)
            mask[to_remove_idx] = False
            # Apply mask to each property in the element
            cleaned_data = data[mask]
            # Create a new PlyElement with the cleaned data
            cleaned_element = PlyElement.describe(cleaned_data, 'vertex')

            # update element
            element.data = cleaned_element.data

    # Save the cleaned PLY data to a new file
    ply_data.write(save_gs_path)

    print("Cleaned point cloud saved to point_cloud_clean.ply")
    
if __name__ == "__main__":
    gs_path = 'output\car1\point_cloud\iteration_30000\point_cloud.ply'
    sorted_indices = get_sort_idx_base_on_color_bright(gs_path)
    start, stop = get_bounds(len(sorted_indices))
    save_gs_path = "output\car1\point_cloud\point_cloud_clean.ply"
    remove_idx_from_ply_file(gs_path, sorted_indices, start, stop, save_gs_path)