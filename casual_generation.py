import torch
import os
import numpy as np
from codes_aux import binvox_rw
from VAEs.vae import VAE

resolution = 120
LATENT_DIM = 40
HIDDEN_DIM = [LATENT_DIM * 4, LATENT_DIM * 2]
INPUT_DIM = resolution ** 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_generations = int(input("Number of casual generation:\n"))
model_path = "Models/vae_mse_kld.pth"                           ##Choose model!
model_name = os.path.splitext(os.path.basename(model_path))[0]
output_dir = os.path.join("causal_designs", model_name)
os.makedirs(output_dir, exist_ok=True)

model = VAE(INPUT_DIM, HIDDEN_DIM, LATENT_DIM).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

with torch.no_grad():
    for i in range(n_generations):
        z = torch.randn(1, LATENT_DIM).to(device)
        generated_flat = model.decode(z)  
        generated = generated_flat.view(1, resolution, resolution, resolution)
        voxel_grid = generated[0].cpu().numpy() > 0.5
        binvox_model = binvox_rw.Voxels(
            voxel_grid,
            voxel_grid.shape,
            translate=[0.0, 0.0, 0.0],
            scale=1.0,
            axis_order='xyz'
        )
        output_path = os.path.join(output_dir, f"generated_{i+1:04d}.binvox")
        with open(output_path, 'wb') as f:
            binvox_model.write(f)
            
print(f"Generated {n_generations} casual design in '{output_dir}'")