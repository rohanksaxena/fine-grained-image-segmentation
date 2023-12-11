import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import xml.etree.ElementTree as ET


class PascalVOCDataset(Dataset):
    def __init__(self):
        self.root_folder = "data/VOCtest_06-Nov-2007/VOCdevkit/VOC2007"
        self.image_folder = os.path.join(self.root_folder, 'JPEGImages')
        self.annotation_folder = os.path.join(self.root_folder, 'Annotations')
        self.images = os.listdir(self.image_folder)
        self.input_size = 224
        self.normalization = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        transforms_list = []
        transforms_list += [
            transforms.Resize([self.input_size, self.input_size], Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(*self.normalization),
        ]
        self.transform = transforms.Compose(transforms_list)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.images[idx])
        image = Image.open(img_name).convert('RGB')

        annotation_name = os.path.join(self.annotation_folder, os.path.splitext(self.images[idx])[0] + '.xml')
        annotation = self.parse_annotation(annotation_name)
        # print(f"get item: {type(image)}")
        if self.transform:
            image = self.transform(image)

        return image, annotation

    def parse_annotation(self, annotation_name):
        tree = ET.parse(annotation_name)
        root = tree.getroot()
        annotation = []

        for obj in root.findall('object'):
            obj_dict = {}
            obj_dict['name'] = obj.find('name').text
            bndbox = obj.find('bndbox')
            obj_dict['xmin'] = int(bndbox.find('xmin').text)
            obj_dict['ymin'] = int(bndbox.find('ymin').text)
            obj_dict['xmax'] = int(bndbox.find('xmax').text)
            obj_dict['ymax'] = int(bndbox.find('ymax').text)
            annotation.append(obj_dict)

        return annotation
