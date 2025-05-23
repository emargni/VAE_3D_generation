import torch
import os
import random
from codes_aux import binvox_rw
import numpy as np
from VAEs.vae import VAE
from Datasets_loader.voxel import VoxelDataset
from torch.utils.data import DataLoader


resolution = 120
LATENT_DIM = 40
PHI=50
HIDDEN_DIM = [LATENT_DIM * 4, LATENT_DIM * 2]
INPUT_DIM = resolution ** 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



models_dir = "Models/"                      ##Add model path
output_base = "guided_generation/"      ##Add output folder path
originals_dir = os.path.join(output_base, "originals")
os.makedirs(output_base, exist_ok=True)
os.makedirs(originals_dir, exist_ok=True)
data_dir='data_vox/120/'                  ##Modify data folder if necessary

dataset = VoxelDataset(data_dir='data_vox/120/')
total_samples = len(dataset)

all_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".binvox")])
class_names = sorted(set(f.split('_')[0] for f in all_files))

print("Available classes:")
for i, cls in enumerate(class_names):
    print(f"{i + 1}. {cls}")

try:
    choice = int(input("\nSelect aircraft class: ")) - 1
    if choice < 0 or choice >= len(class_names):
        raise ValueError
    class_name = class_names[choice]
    x=(choice)*20

    class_files = [f for f in all_files if f.startswith(class_name)]
    print(f"\nAvailable design for '{class_name}':")
    for i, fname in enumerate(class_files):
        print(f"{i + 1}. {fname}")

    file_choice = int(input("\nSelect aircraft nr: ")) - 1
    if file_choice < 0 or file_choice >= len(class_files):
        raise ValueError
    selected_filename = class_files[file_choice]
    print(file_choice)
    x+= file_choice

    repetitions = int(input("How many new designs: "))
except ValueError:
    print("Not valid input.")
    
    exit()

random_indices= [x]


original_samples = [dataset[i].unsqueeze(0) for i in random_indices]

print("\nDesign extraction\n")
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
            for j in range(10):
                for idx, data in enumerate(original_samples):
                    data = data.to(device)
                    data_flat = data.view(data.size(0), -1)

                    hidden = model.encoder(data_flat)
                    mu = model.fc_mu(hidden)
                    logvar = model.fc_logvar(hidden)
                    std = torch.exp(0.5 * logvar)
                    eps = torch.randn_like(std)
                    z = mu  + random.randint(0, PHI)*std
                    dec_input = model.decoder_input(z)
                    recon_flat = model.decoder(dec_input)

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

                    output_path = os.path.join(output_dir, f"{model_name}_var{j}_recon_{idx+1}.binvox")
                    with open(output_path, 'wb') as f:
                        binvox_model.write(f)

        print(f"new design saved in '{output_dir}'")
