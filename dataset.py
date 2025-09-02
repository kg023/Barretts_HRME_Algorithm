import HRMECrop
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import pandas as pd
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torch.utils.data._utils.collate import default_collate
from sklearn.model_selection import GroupShuffleSplit

class CustomCrop:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.cache = {}

    def __call__(self, sample):
        valid_subregions = self.cache.get(sample["image_filepath"])
        if valid_subregions is None:
            valid_subregions = HRMECrop.find_valid_rotated_subregions(sample["mask_filepath"], self.width, self.height)
        idx = np.random.randint(0, len(valid_subregions))
        valid_subregion = valid_subregions[idx]
        sample["cx"], sample["cy"], sample["angle"], sample["width"], sample["height"] = valid_subregion
        img = HRMECrop.get_rotated_rect_crop(sample["image_filepath"], sample["cx"], sample["cy"], sample["angle"], 
                                             sample["width"], sample["height"])
        sample["bbox"], sample["rotated_rect_corners"] = HRMECrop.get_rotated_rect_attributes(sample["cx"], sample["cy"], 
                                                                                              sample["angle"], sample["width"], 
                                                                                              sample["height"])
        return img
    

class BarrettsHRMEDataset(Dataset):
    def __init__(self, annotations_csv_filepath: Path, data_partition: str = "train", return_sample_data=True):
        annotations_df = pd.read_csv(annotations_csv_filepath)
        self.custom_crop = CustomCrop(350, 350)
        self.transforms = transforms.Compose([transforms.Resize(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                                   std=[0.229, 0.224, 0.225])])
        
        splitter = GroupShuffleSplit(test_size=0.3, n_splits=1, random_state=42)
        for train_idx, test_idx in splitter.split(annotations_df, 
                                                  groups=annotations_df['Subject_ID'], 
                                                  y=annotations_df['Local_Pathology_Final']):
            self.annotations_df_train = annotations_df.iloc[train_idx].reset_index(drop=True)
            self.annotations_df_test = annotations_df.iloc[test_idx].reset_index(drop=True)
        
        self.data_partition = data_partition
        self.return_sample = return_sample_data

    def __len__(self):
        if self.data_partition=="train":
            return len(self.annotations_df_train)
        elif self.data_partition=="test":
            return len(self.annotations_df_test)
    
    def __getitem__(self, idx):
        if self.data_partition=="train":
            img_filepath = self.annotations_df_train.loc[idx, "Lesion_HRME_Image_Filepath"]
            mask_filepath = self.annotations_df_train.loc[idx, "Lesion_HRME_Mask_Filepath"]

            sample = {"image_filepath": img_filepath, "mask_filepath": mask_filepath}

            cropped_img = self.custom_crop(sample)
            cropped_img = Image.fromarray(cropped_img).convert("RGB")
            
            # Apply transforms to image
            img = self.transforms(cropped_img)

            label = torch.tensor(self.annotations_df_train.loc[idx, "Local_Pathology_Final"])
        
        elif self.data_partition=="test":
            img_filepath = self.annotations_df_test.loc[idx, "Lesion_HRME_Image_Filepath"]
            mask_filepath = self.annotations_df_test.loc[idx, "Lesion_HRME_Mask_Filepath"]

            sample = {"image_filepath": img_filepath, "mask_filepath": mask_filepath}

            cropped_img = self.custom_crop(sample)
            cropped_img = Image.fromarray(cropped_img).convert("RGB")

            img = self.transforms(cropped_img)

            label = torch.tensor(self.annotations_df_test.loc[idx, "Local_Pathology_Final"])

        if self.return_sample:
            return img, label, sample
        else:
            return img, label
    

def collate_with_samples(batch):
    imgs, labels, samples = zip(*batch)
    imgs = default_collate(imgs)
    labels = default_collate(labels)
    return imgs, labels, list(samples)

def create_dataloader(dataset: BarrettsHRMEDataset, batch_size: int = 4):
    if dataset.data_partition=="train":
        class_counts = torch.bincount(torch.tensor(dataset.annotations_df_train["Local_Pathology_Final"].values))
        class_weights = 1.0/class_counts.float()
        sample_weights = class_weights[torch.tensor(dataset.annotations_df_train["Local_Pathology_Final"].values)]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        if dataset.return_sample:
            dataloader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler, num_workers=0, 
                                    collate_fn=collate_with_samples)
        else:
            dataloader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler, num_workers=0)

    elif dataset.data_partition=="test":
        if dataset.return_sample:
            dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=0, 
                                    collate_fn=collate_with_samples)
        else:
            dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=0)
    
    return dataloader