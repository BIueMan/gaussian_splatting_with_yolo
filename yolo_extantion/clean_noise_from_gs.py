import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB

gs_path = 'output\plant_2\point_cloud\iteration_30000\point_cloud.ply'

plydata = PlyData.read(gs_path)

xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"])),  axis=1)
features_dc = np.zeros((xyz.shape[0], 3, 1))
features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
rgb = SH2RGB(features_dc[..., 0])
# clamp the lower bound of rgb values to 0
rgb = np.maximum(rgb, 0)
# norm
rgb = rgb / np.max(rgb)

# Take the absolute value of each color component
abs_color_matrix = np.sum(rgb, axis = 1)

# Sort the colors by the sum of their RGB components
sorted_colors = abs_color_matrix[np.argsort(abs_color_matrix)]


# Sort the distances and keep the original indices
sorted_indices = np.argsort(abs_color_matrix)
sorted_distances = abs_color_matrix[sorted_indices]

# Plot the sorted distances
plt.figure(figsize=(10, 6))
plt.plot(sorted_distances)
plt.title('Distances of Points from Mean Location')
plt.xlabel('Point Index (Sorted)')
plt.ylabel('Distance from Mean')
plt.grid(True)
plt.show()

# Identify points to remove
to_remove_idx = sorted_indices[:4000]

# Load the PLY file using PlyData to manipulate properties
ply_data = PlyData.read("output/plant_2/point_cloud/iteration_30000/point_cloud.ply")

# For each property, remove the elements in to_remove_idx
for element in ply_data.elements:
    if element.name == 'vertex':
        # Convert element data to a structured array for easier manipulation
        data = element.data

        # Create a mask to keep only the desired indices
        mask = np.ones(data.shape[0], dtype=bool)
        mask[to_remove_idx] = False

        # Apply mask to each property in the element
        cleaned_data = data[mask]

        # Create a new PlyElement with the cleaned data
        cleaned_element = PlyElement.describe(cleaned_data, 'vertex')

        # Replace the original element with the cleaned one
        element.data = cleaned_element.data

# Save the cleaned PLY data to a new file
ply_data.write("output/point_cloud_clean.ply")

print("Cleaned point cloud saved to point_cloud_clean.ply")