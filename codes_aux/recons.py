import torch
import os
import random
import binvox_rw
import numpy as np
from VAEs.vae import VAE
from Datasets_loader.voxel import VoxelDataset
from torch.utils.data import DataLoader


resolution = 120
LATENT_DIM = 2
HIDDEN_DIM = [LATENT_DIM * 4, LATENT_DIM * 2]
INPUT_DIM = resolution ** 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random_seed = 42
random.seed(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)

models_dir = "Models/"
output_base = "ricostruzioni_latent/vae_mse_kld.pth"
originals_dir = os.path.join(output_base, "originals")
os.makedirs(output_base, exist_ok=True)
os.makedirs(originals_dir, exist_ok=True)

dataset = VoxelDataset(data_dir='data_vox/120/')
total_samples = len(dataset)
random_indices = random.sample(range(total_samples), 10)
print(random_indices)

original_samples = [dataset[i].unsqueeze(0) for i in random_indices]

print("\n10 causal design extraction\n")
for idx, data in enumerate(original_samples):
    voxel_grid = data[0].numpy() > 0.5
    dims = voxel_grid.shape
    translate = [0.0, 0.0, 0.0]
    scale = 1.0
    axis_order = 'xyz'

    binvox_model = binvox_rw.Voxels(
        voxel_grid,
        dims,
        translate,
        scale,
        axis_order
    )

    output_path = os.path.join(originals_dir, f"original_{idx+1}.binvox")
    with open(output_path, 'wb') as f:
        binvox_model.write(f)

print("original design saved/'")


for model_file in os.listdir(models_dir):
    if model_file.endswith(".pth"):
        model_path = os.path.join(models_dir, model_file)
        model_name = os.path.splitext(model_file)[0]

        print(f"\nmodel gen.: {model_name}")

        output_dir = os.path.join(output_base, model_name)
        os.makedirs(output_dir, exist_ok=True)

        
        model = VAE(INPUT_DIM, HIDDEN_DIM, LATENT_DIM).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        with torch.no_grad():
            for idx, data in enumerate(original_samples):
                data = data.to(device)
                data_flat = data.view(data.size(0), -1)
                
                
                recon_flat, _, _ = model(data_flat)
                recon = recon_flat.view(data.size(0), resolution, resolution, resolution)

                voxel_grid = recon[0].cpu().numpy() > 0.5
                dims = voxel_grid.shape
                translate = [0.0, 0.0, 0.0]
                scale = 1.0
                axis_order = 'xyz'

                binvox_model = binvox_rw.Voxels(
                    voxel_grid,
                    dims,
                    translate,
                    scale,
                    axis_order
                )

                output_path = os.path.join(output_dir, f"{model_name}_recon_{idx+1}.binvox")
                with open(output_path, 'wb') as f:
                    binvox_model.write(f)

        print(f"reconstruction saved in '{output_dir}'")
