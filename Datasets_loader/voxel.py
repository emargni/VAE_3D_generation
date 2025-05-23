import os
import torch
from torch.utils.data import Dataset
import numpy as np

class VoxelDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.binvox')]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        voxel_data = self.read_binvox(file_path)
        voxel_tensor = torch.tensor(voxel_data, dtype=torch.float32)
        if self.transform:
            voxel_tensor = self.transform(voxel_tensor)
        return voxel_tensor

    def read_binvox(self, filepath):
        with open(filepath, 'rb') as f:
            line = f.readline().strip()
            if not line.startswith(b'#binvox'):
                raise ValueError("Not a binvox file")
            dims = None
            translate = None
            scale = None
            while True:
                line = f.readline().strip()
                if line.startswith(b'dim'):
                    dims = list(map(int, line.split()[1:]))
                elif line.startswith(b'translate'):
                    translate = list(map(float, line.split()[1:]))
                elif line.startswith(b'scale'):
                    scale = float(line.split()[1])
                elif line == b'data':
                    break
            raw_data = np.frombuffer(f.read(), dtype=np.uint8)
            values = []
            i = 0
            while i < len(raw_data):
                value = raw_data[i]
                count = raw_data[i+1]
                values.extend([value] * count)
                i += 2
            voxel = np.array(values, dtype=bool).reshape(dims)
            return voxel
