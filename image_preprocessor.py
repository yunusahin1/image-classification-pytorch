import os
import pandas as pd
import cv2
from typing import List, Tuple, Dict
import torchvision.transforms as transforms
import torch
from PIL import Image
from sklearn.model_selection import train_test_split

def label_images(prefix: str = './data/animals/') -> pd.DataFrame:
    label_json = {'file_name': [], 'label': []}
    folders = os.listdir(prefix)
    for folder in folders:
        for file in os.listdir(prefix + folder):
            if file.endswith('.jpg'):
                label_json['file_name'].append(prefix + folder + '/' + file)
                label_json['label'].append(folder)
    label_df = pd.DataFrame(label_json)
    return label_df

def image_to_tensors(df: pd.DataFrame) -> Tuple[torch.Tensor, List]:
    image_files = df['file_name']
    labels = df['label']
    image_tensors = []
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    for file in image_files:
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image.astype('uint8'))
        tensor = transform(pil_image)
        image_tensors.append(tensor)

    stacked_tensors = torch.stack(image_tensors)
    return stacked_tensors, labels.tolist()

def data_augmentation(images: torch.Tensor) -> torch.Tensor:
    augmentation_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(45),
    ])
    
    normalization = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
    
    augmented_images = []
    for img in images:
        pil_img = transforms.ToPILImage()(img)
        augmented_img = augmentation_transform(pil_img)
        tensor_img = transforms.ToTensor()(augmented_img)
        normalized_img = normalization(tensor_img)
        augmented_images.append(normalized_img)

    all_images = torch.cat([images, torch.stack(augmented_images)], dim=0)
    return all_images

def create_train_test_split(images: torch.Tensor, labels: List, test_size: float = 0.2, random_state: int = 42) -> Dict[str, object]:
    
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    return {
        'train_images': X_train,
        'test_images': X_test,
        'train_labels': y_train,
        'test_labels': y_test
    }
