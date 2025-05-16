import os
import pandas as pd
import cv2
from typing import List, Tuple
import torchvision.transforms as transforms

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

def image_to_tensors(df: pd.DataFrame) -> Tuple[List, List]:
    image_files = df['file_name']
    labels = df['label']
    image_tensors = []
    for file in image_files:
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0
        image_tensors.append(image)
    return image_tensors, labels.tolist()

def data_augmentation(image: List) -> List:
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(45),
        transforms.ToTensor()
    ])
    augmented_images = []
    for img in image:
        img = transform(img)
        augmented_images.append(img)

    all_images = image + augmented_images
    return all_images


