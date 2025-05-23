
import os
import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from VAEs.vae import VAE
from Datasets_loader.voxel import VoxelDataset

models_dir = "Models/"
model_file = "vae_mse_kld.pth"          ##Choose model`!`
resolution = 120                        ##Put model hyperparameters
latent_dim = 40                         ##Put model hyperparameters
hidden_dim = [latent_dim * 4, latent_dim * 2]
input_dim = resolution ** 3
max_samples = 180                       ##Put number of total input designs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = os.path.join(models_dir, model_file)
model = VAE(input_dim, hidden_dim, latent_dim).to(device)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint)
model.eval()

data_folder = f"data_vox/{resolution}/"
filenames = sorted([f for f in os.listdir(data_folder) if f.endswith('.binvox')])
dataset = VoxelDataset(data_dir=data_folder)
total = min(len(filenames), len(dataset), max_samples)
indices = list(range(total))

latents = []
categories = []
with torch.no_grad():
    for idx in indices:
        x = dataset[idx].unsqueeze(0).to(device)
        x_flat = x.view(1, -1)
        _, mu, _ = model(x_flat)
        latents.append(mu.cpu().numpy().reshape(-1))
        filename = filenames[idx]
        category = filename.split('_')[0]
        categories.append(category)

latents = np.vstack(latents)

pca = PCA(n_components=3)
latents_3d = pca.fit_transform(latents)

unique_cats = sorted(set(categories))
colormap = plt.cm.get_cmap('tab10', len(unique_cats))
colors = [colormap(unique_cats.index(cat)) for cat in categories]

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(
    latents_3d[:,0], latents_3d[:,1], latents_3d[:,2],
    c=colors, s=20, depthshade=True
)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

handles = [plt.Line2D([0],[0], marker='o', color='w',
            markerfacecolor=colormap(i), markersize=8)
           for i in range(len(unique_cats))]
ax.legend(handles, unique_cats, title='Aircraft model')

plt.tight_layout()
plt.savefig('latent_space.png', dpi = 1000)
plt.show()

