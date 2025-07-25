# CT SYNTHETIC SWEEP POSE ESTIMATION
# created 07/07/2025

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import os

########## DATASET CLASS ##########

class CTSweepDataset(Dataset):
    """
    Dataset: Python wrapper that takes a list of filepaths to CT images and loads the image + label for training.
    """
    def __init__(self, img_paths, angle_labels, tform=None):
        """
        Initializes Dataset class
        Inputs:
            img_paths: list of filepaths to images -> list[str]
            angle_labels: list of corresponding angles -> list[float]
            tform: torchvision transforms (i.e. for resizing, normalization)
        """
        assert len(img_paths) == len(angle_labels), "Mismatch in dataset size"
        self.img_paths = img_paths
        self.angle_labels = angle_labels
        self.transform = tform or T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        angle = self.angle_labels[idx]

        image = Image.open(img_path).convert("L")  # Grayscale

        if self.transform:
            image = self.transform(image)

        angle = torch.tensor(angle, dtype=torch.float32)

        return image, angle


########## PARSE FOR DATASET ##########

def parse_dataset(folder_path, rotation):
    """
    Parses folder with ct sweep datasets for filepath and angle labels.
    """
    img_paths = []
    angle_labels = []

    for patient_folder in sorted(os.listdir(folder_path)):
        patient_path = os.path.join(folder_path, patient_folder)
        rotation_path = os.path.join(patient_path, rotation)

        if not os.path.isdir(rotation_path):
            continue

        for fname in sorted(os.listdir(rotation_path)):
            if fname.endswith(".png"):
                angle_str = os.path.splitext(fname)[0]  # 'rotated_+00'
                angle = float(angle_str.replace("rotated_", ""))  # '00' → 0.0
                img_paths.append(os.path.join(rotation_path, fname))
                angle_labels.append(angle)
    
    print(f"Parsed {len(img_paths)} images from {folder_path}")
    return img_paths, angle_labels

########## SAMPLE USAGE ##########

# img_paths, angle_labels = parse_dataset("/Users/emiz/Desktop/ct_sweep_datasets/testing", "roll")