import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from skimage.color import rgb2lab
from superpixel import Superpixel
from model import SSN_DINO
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):

    def __init__(self, name, mode="test"):
        if name == "CUB":
            if mode == "test":
                self.flag = "0"
            else:
                self.flag = "1"

            self.root = "data/CUB/CUB_200_2011/"
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

            for id in self.images_subset:
                self.images.append(self.image_paths[id])

        elif name == "ECSSD":
            self.root = "data/ECSSD-data/"
            self.images_root = "images/images"
            self.masks_root = "ground_truth_mask/ground_truth_mask"
            image_list = os.listdir(os.path.join(self.root, self.images_root))
            self.images = [x.split(".")[0] for x in image_list]

        elif name == "DUTS":
            self.root = "data/DUTS/DUTS-TE/"
            self.images_root = "DUTS-TE-Image"
            self.masks_root = "DUTS-TE-Mask"
            image_list = os.listdir(os.path.join(self.root, self.images_root))
            self.images = [x.split(".")[0] for x in image_list]

        elif name == "DUT-OMRON":
            self.root = "data/DUT-OMRON/"
            self.images_root = "DUT-OMRON-image"
            self.masks_root = "pixelwiseGT-new-PNG"
            image_list = os.listdir(os.path.join(self.root, self.images_root))
            self.images = [x.split(".")[0] for x in image_list]

        self.image_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        self.mask_transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load the image and mask
        image = Image.open(os.path.join(self.root, self.images_root, self.images[idx]) + ".jpg")
        mask = Image.open(os.path.join(self.root, self.masks_root, self.images[idx]) + ".png")

        # Convert image to RGB and apply transformations
        image = image.convert('RGB')
        image = self.image_transform(image)

        # Convert mask to grayscale to preserve the class labels
        mask = mask.convert('L')  # 'L' mode converts the image to grayscale
        mask_np = np.array(mask)

        # Apply transformation to mask (no normalization here as it's a class map, not an image)
        mask = self.mask_transform(mask_np)

        return image, mask, os.path.join(self.root, self.images_root, self.images[idx]) + ".jpg", os.path.join(
            self.root, self.masks_root, self.images[idx] + ".png"), self.images[idx]
