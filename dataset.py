import torchvision.transforms.functional as F
import numpy as np
import random
import os
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from torchvision import transforms


class ToTensor(object):
    def __call__(self, data):
        image, label = data['image'], data['label']
        # Image: convert to tensor [C, H, W] with values in [0, 1]
        # Label: convert to tensor, add channel dimension, and convert to float
        return {
            'image': F.to_tensor(image),
            'label': torch.from_numpy(np.array(label)).float().unsqueeze(0)  # [H, W] -> [1, H, W]
        }

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        image, label = data['image'], data['label']
        
        # Resize image normally
        resized_image = F.resize(image, self.size)
        
        # CHANGED: Resize label with NEAREST interpolation to preserve class indices
        # PIL Image needs to be converted to tensor, resized, then back
        label_tensor = torch.from_numpy(np.array(label)).unsqueeze(0).float()
        resized_label = F.resize(label_tensor, self.size, interpolation=InterpolationMode.NEAREST)
        resized_label = resized_label.squeeze(0).numpy().astype(np.uint8)
        resized_label = Image.fromarray(resized_label, mode='L')
        
        return {'image': resized_image, 'label': resized_label}


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label = data['image'], data['label']

        if random.random() < self.p:
            return {'image': F.hflip(image), 'label': F.hflip(label)}

        return {'image': image, 'label': label}


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label = data['image'], data['label']

        if random.random() < self.p:
            return {'image': F.vflip(image), 'label': F.vflip(label)}

        return {'image': image, 'label': label}


class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # Only normalize the image, not the label (label contains class indices)
        image = F.normalize(image, self.mean, self.std)
        return {'image': image, 'label': label}
    

class FullDataset(Dataset):
    def __init__(self, image_root, gt_root, size, mode):
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.size = size
        
        if mode == 'train':
            self.transform = transforms.Compose([
                Resize((size, size)),
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                ToTensor(),
                Normalize()
            ])
        else:
            self.transform = transforms.Compose([
                Resize((size, size)),
                ToTensor(),
                Normalize()
            ])

    def __getitem__(self, idx):
        image = self.rgb_loader(self.images[idx])
        label = self.binary_loader(self.gts[idx])  # Still grayscale, but now contains class indices
        data = {'image': image, 'label': label}
        data = self.transform(data)
        return data

    def __len__(self):
        return len(self.images)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        # CHANGED: Still loads as grayscale, but now values are 0-6 (class indices)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
        

class TestDataset:
    def __init__(self, image_root, gt_root, size):
        # Get all image files
        image_files = sorted([f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')])
        gt_files = sorted([f for f in os.listdir(gt_root) if f.endswith('.png')])
        
        # Match images with their corresponding masks
        self.images = []
        self.gts = []
        
        for img_file in image_files:
            # Try to find corresponding mask (handle different extensions)
            base_name = os.path.splitext(img_file)[0]
            mask_file = base_name + '.png'
            
            if mask_file in gt_files:
                self.images.append(os.path.join(image_root, img_file))
                self.gts.append(os.path.join(gt_root, mask_file))
            else:
                print(f"Warning: No mask found for {img_file}")
        
        self.size = len(self.images)
        self.index = 0
        
        print(f"Loaded {self.size} image-mask pairs")
        
        # Verify we have data
        if self.size == 0:
            raise ValueError("No matching image-mask pairs found!")
        
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def load_data(self):
        # Check if index is within bounds
        if self.index >= self.size:
            raise StopIteration("All images have been processed")
        
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)

        # Load as class indices
        gt = self.binary_loader(self.gts[self.index])
        gt = np.array(gt)  # Contains values 0-6

        name = os.path.basename(self.images[self.index])

        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
