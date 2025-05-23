import os
from codes_aux import binvox_rw
import numpy as np
import pandas as pd
from scipy.ndimage import label, sum as ndi_sum

def load_binvox(filepath):
    with open(filepath, 'rb') as f:
        model = binvox_rw.read_as_3d_array(f)
    return model.data

def compute_connectivity_percentage(voxel_grid):
    structure = np.ones((3, 3, 3), dtype=int)
    labeled, num_features = label(voxel_grid, structure=structure)
    if num_features == 0:
        return 0.0
    sizes = ndi_sum(voxel_grid, labeled, index=range(1, num_features + 1))
    largest_component_size = sizes.max()
    total_voxels = voxel_grid.sum()
    return (largest_component_size / total_voxels) * 100

if __name__ == "__main__":
    input_folder = "causal_designs/vae_mse_kld"             ##Choose folder with designs to filter              ##Modify casual designs folder if necessary

    threshold=input("Connectivity treshold:\n")
    

    results = []
    for filename in os.listdir(input_folder):
        if filename.endswith(".binvox"):
            filepath = os.path.join(input_folder, filename)
            voxel_grid = load_binvox(filepath)
            percentage = compute_connectivity_percentage(voxel_grid)
            results.append({"filename": filename, "connectivity_percentage": percentage})
            if percentage < threshold:
                os.remove(filepath)
    df = pd.DataFrame(results)
    df.to_csv("connectivity_scores.csv", index=False)
