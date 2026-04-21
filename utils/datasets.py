import os
import cv2
import torch
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader
import albumentations as A
from collections import Counter
from tqdm import tqdm

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

class BaseTransform:
    def __init__(self, size=384, is_training=True):
        self.size = size
        if is_training:
            self.transform = A.Compose([
                A.Resize(height=size, width=size),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.7),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                A.Blur(blur_limit=3, p=0.3),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=size, width=size),
            ])

    def __call__(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Bad Image")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=img)
        img = augmented['image']
        
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
        return img_tensor

class RicePestDataset(datasets.ImageFolder):
    def __init__(self, root, is_training=True, image_size=384, min_samples_per_class=5):
        self.transform_logic = BaseTransform(size=image_size, is_training=is_training)
        
        super(RicePestDataset, self).__init__(root, transform=None)
        
        print(f"Scanning dataset at {root}...")
        self.samples = self._clean_dataset(self.samples, min_samples_per_class)
        self.targets = [s[1] for s in self.samples]
        self.imgs = self.samples
        self.classes = self.classes 

    def _clean_dataset(self, samples, min_samples):
        clean_samples = []
        valid_indices = []
        
        for idx, (path, target) in enumerate(tqdm(samples, desc="Verifying Images")):
            try:
                with open(path, 'rb') as f:
                    check_chars = f.read(4)
                    if not check_chars:
                        continue
                
                img = cv2.imread(path)
                if img is not None and img.size > 0:
                    clean_samples.append((path, target))
            except:
                continue
        
        class_counts = Counter([s[1] for s in clean_samples])
        final_samples = []
        skipped_classes = []
        
        for path, target in clean_samples:
            if class_counts[target] >= min_samples:
                final_samples.append((path, target))
            else:
                if target not in skipped_classes:
                    skipped_classes.append(target)
        
        removed_count = len(samples) - len(final_samples)
        print(f"Dataset Cleaned: Removed {removed_count} corrupted/small-class images.")
        if skipped_classes:
            print(f"Skipped {len(skipped_classes)} classes due to insufficient samples.")
            
        return final_samples

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.transform_logic(path)
        return sample, target

def calculate_class_weights(dataset, device):
    counts = Counter(dataset.targets)
    classes = sorted(counts.keys())
    
    if not classes:
        return None

    num_samples = len(dataset)
    num_classes = len(classes)
    
    weight_list = []
    for i in range(num_classes):
        count = counts.get(i, 0)
        if count > 0:
            w = num_samples / (num_classes * count)
        else:
            w = 0.0 
        weight_list.append(w)
        
    weights = torch.FloatTensor(weight_list).to(device)
    return weights

def get_dataloaders(config):
    data_root = config['data_root']
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    img_size = config['input_size']
    
    train_dir = os.path.join(data_root, 'train')
    valid_dir = os.path.join(data_root, 'valid')
    test_dir = os.path.join(data_root, 'test')
    
    train_ds = RicePestDataset(train_dir, is_training=True, image_size=img_size)
    valid_ds = RicePestDataset(valid_dir, is_training=False, image_size=img_size)
    
    if os.path.exists(test_dir):
        test_ds = RicePestDataset(test_dir, is_training=False, image_size=img_size)
    else:
        test_ds = valid_ds

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, 
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, pin_memory=True)
    
    return train_loader, valid_loader, test_loader