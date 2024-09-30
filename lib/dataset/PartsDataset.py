import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class PartsDataset(Dataset):

    def __init__(self, name, mode="test"):
        if name == "CUB":
            if mode == "test":
                self.flag = "0"
            else:
                self.flag = "1"

            self.root = "data/CUB/CUB_200_2011/"
            self.num_kps = 15
            self.images_root = "images"
            self.masks_root = "segmentations"
            self.parts = "parts"
            self.image_paths = {}
            self.images = []
            with open(os.path.join(self.root, "images.txt")) as f:
                for line in f.readlines():
                    id, path = line.split(" ")
                    path = os.path.splitext(path)[0]
                    self.image_paths[id] = path.strip()

            self.images_subset = []
            with open(os.path.join(self.root, "train_test_split.txt")) as f:
                for line in f.readlines():
                    id, split = line.split(" ")
                    if split.strip() == self.flag:
                        self.images_subset.append(id)

            # Load part information
            self.parts_info = np.genfromtxt(os.path.join(self.root, self.parts, 'part_locs.txt'), dtype=float)

            for id in self.images_subset:
                self.images.append((id, self.image_paths[id]))

        self.image_transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        self.mask_transform = T.Compose([T.ToTensor()])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.root, self.images_root, self.images[idx][1]) + ".jpg")
        mask = Image.open(os.path.join(self.root, self.masks_root, self.images[idx][1]) + ".png")
        image = image.convert('RGB')  # to reproduce (???)
        mask = mask.convert('RGB')  # to reproduce (???)
        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        mask = (mask > 0.5)[0].long()  # TODO: this could be improved
        id = int(self.images[idx][0])
        parts = self.parts_info[id * self.num_kps - self.num_kps: id * self.num_kps][:, 1:]
        visible = parts[:, 3] > 0.5
        parts = parts[visible]
        return image, mask, os.path.join(self.root, self.images_root, self.images[idx][1]) + ".jpg", os.path.join(
            self.root, self.masks_root, self.images[idx][1] + ".png"), self.images[idx][1], id, parts
