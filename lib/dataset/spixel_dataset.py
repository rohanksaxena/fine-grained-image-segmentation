from torch.utils.data import Dataset
import torch

class SpixelDataset(Dataset):
    def __init__(self, spixel_list):
        self.x1, self.x2 = [], []
        self.spixel_list = spixel_list
        for key, val in spixel_list.items():
            for neighbor in val.neighbor_spixels:
                self.x1.append(key)
                self.x2.append(neighbor)

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, idx):
        return torch.squeeze(self.spixel_list[self.x1[idx]].features), torch.squeeze(self.spixel_list[self.x2[idx]].features)
