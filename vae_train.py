import os
import random
import torch
from codes_aux import binvox_rw
import numpy as np
from torch.utils.data import DataLoader
from VAEs.vae import VAE
from sklearn.decomposition import PCA
from Loss.loss_general import (
    loss_mse_kld, loss_iou_kld, loss_bce_kld,
    loss_dice_kld, loss_focal_kld,
    )
from Datasets_loader.voxel import VoxelDataset
from codes_aux import converter_dimensions
import matplotlib.pyplot as plt

input_folder = 'data_obj/'                                  ##Input data folder
resolution = int(input("Resolution:\n"))
output_folder = f'data_vox/{resolution}'

MAX_EPOCHS = int(input("Epochs:\n"))

BATCH_SIZE = 18
LEARNING_RATE = 1e-4
LATENT_DIM = int(input("Latent space dimensions:\n"))
recc=input("Reconstructions? (y/n)\n")
HIDDEN_DIM = [LATENT_DIM * 4, LATENT_DIM * 2]
INPUT_DIM = resolution ** 3
models_dir = 'Models/'                                          ##Folder where the model will be saved


def main():
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    if input("Voxelize? (y/n)\n").lower() == 'y':
        converter_dimensions.process_folder(input_folder, output_folder, resolution)

    
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    
    dataset = VoxelDataset(data_dir=output_folder)
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,    
        pin_memory=True,
    )

    loss_functions = {                                          ##Select or deselect desired loss functions
        'mse_kld': loss_mse_kld,
        'iou_kld': loss_iou_kld,
        #'bce_kld': loss_bce_kld,
        #'dice_kld': loss_dice_kld,
        #'focal_kld': loss_focal_kld,
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    for loss_name, loss_fn in loss_functions.items():
        print(f"\nTraining with {loss_name}")
        model = VAE(INPUT_DIM, HIDDEN_DIM, LATENT_DIM).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        epoch_losses = []

        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            running_loss = 0.0
            for data in train_loader:
                data = data.to(device)
                data_flat = data.view(data.size(0), -1)
                optimizer.zero_grad()
                recon_flat, mu, logvar = model(data_flat)
                loss = loss_fn(recon_flat, data_flat, mu, logvar, INPUT_DIM)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * data.size(0)
            avg_loss = running_loss / len(train_loader.dataset)
            epoch_losses.append(avg_loss)
            print(f"Epoch {epoch}/{MAX_EPOCHS}: {avg_loss:.6f}")

        
        torch.save(model.state_dict(), os.path.join(models_dir, f"vae_{loss_name}.pth"))
        """
        plt.figure()
        plt.plot(range(1, MAX_EPOCHS + 1), epoch_losses)
        plt.title(f"Loss: {loss_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        #plt.savefig(os.path.join(models_dir, f"loss_{loss_name}.png"), dpi=300)
        #plt.close()
        """
        print(f"Saved vae_{loss_name}.pth and loss_{loss_name}.png")

        if recc.lower() == 'y':

            output_base = f"recostruction_latent/vae_{loss_name}.pth"
            originals_dir = os.path.join(output_base, "originals")
            os.makedirs(output_base, exist_ok=True)
            os.makedirs(originals_dir, exist_ok=True)

            dataset = VoxelDataset(data_dir=f'data_vox/{resolution}/')
            total_samples = len(dataset)
            random_indices = random.sample(range(total_samples), 10)

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
        

                    


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  
    main()
    
