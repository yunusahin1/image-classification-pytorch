import torch
from image_preprocessor import label_images, image_to_tensors, data_augmentation
from utils import config_yaml

def main() -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    config = config_yaml()
    
    label_df = label_images()
    image_tensors, labels = image_to_tensors(label_df)
    

    if config.get("preprocessing", {}).get("augmentation", False):
        image_tensors = data_augmentation(image_tensors)
        original_len = len(labels)
        labels = labels + labels[:original_len]


if __name__ == "__main__":
    main()
