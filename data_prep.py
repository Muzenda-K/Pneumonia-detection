from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch

# ================
# Helper Functions
# ================

def plot_samples(dataset_split, num_samples=5):
    plt.figure(figsize=(15, 5))
    indices = {0: [], 1: []}
    
    for idx, sample in enumerate(dataset_split):
        if len(indices[sample['label']]) < num_samples:
            indices[sample['label']].append(idx)

    for i, idx in enumerate(indices[0]):
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(dataset_split[idx]['image'], cmap='gray')
        plt.title(f"Normal - Label: {dataset_split[idx]['label']}")
        plt.axis('off')

    for i, idx in enumerate(indices[1]):
        plt.subplot(2, num_samples, num_samples + i + 1)
        plt.imshow(dataset_split[idx]['image'], cmap='gray')
        plt.title(f"Pneumonia - Label: {dataset_split[idx]['label']}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def plot_class_distribution(dataset_split, split_name):
    labels = np.array(dataset_split['label'])
    class_counts = np.bincount(labels)
    classes = ['Normal', 'Pneumonia']

    plt.figure(figsize=(8, 5))
    bars = plt.bar(classes, class_counts, color=['#4CAF50', '#F44336'], width=0.6, edgecolor='black')
    
    plt.title(f'Class Distribution - {split_name} Set', fontsize=14, pad=20)
    plt.ylabel('Number of Images', fontsize=12)
    plt.xlabel('Diagnosis', fontsize=12)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:,}\n({height/len(labels):.1%})',
                 ha='center', va='bottom', fontsize=11)

    plt.ylim(0, max(class_counts) * 1.15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def check_image_channels(dataset_split):
    for idx, sample in enumerate(dataset_split):
        image = sample['image']
        print(f"Sample {idx}: Image Shape - {image.size}, Mode - {image.mode}")
        if idx >= 5:
            break


def plot_pixel_intensity_histograms(dataset_split, num_samples=5):
    plt.figure(figsize=(15, 5))
    for label in [0, 1]:
        class_images = []
        for idx, sample in enumerate(dataset_split):
            if sample['label'] == label and len(class_images) < num_samples:
                image = sample['image']
                class_images.append(np.array(image).flatten())

        for i, pixels in enumerate(class_images):
            plt.subplot(2, num_samples, label * num_samples + i + 1)
            plt.hist(pixels, bins=256, color='skyblue', alpha=0.7)
            plt.title(f"Class {label} - Sample {i+1}")
            plt.xlabel("Pixel Intensity")
            plt.ylabel("Frequency")
            plt.xlim(0, 255)

    plt.tight_layout()
    plt.show()


def plot_single_image_histogram(sample_idx, dataset_split):
    sample = dataset_split[sample_idx]
    image = sample['image']
    pixels = np.array(image).flatten()

    plt.figure(figsize=(8, 4))
    plt.hist(pixels, bins=256, color='skyblue', alpha=0.7)
    plt.title(f"Image {sample_idx} - Class {sample['label']} - Pixel Intensity Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.xlim(0, 255)
    plt.show()


# =====================
# Data Preparation Code
# =====================
# Define collate function at top level
def custom_collate(batch):
    batch = [item for item in batch if item[0].ndim == 3]
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    return images, labels


class ChestXRayDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        idx = int(idx)
        item = self.hf_dataset[idx]
        image = item['image'].convert("L")  # Grayscale

        if self.transform:
            image = self.transform(image)

        return image, item['label']


def get_dataloaders(batch_size=32):
    dataset = load_dataset("hf-vision/chest-xray-pneumonia")

    # Define transforms
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),

        # Augmentation
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomHorizontalFlip(p=0.5),
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    train_dataset = ChestXRayDataset(dataset['train'], transform=transform_train)
    val_dataset = ChestXRayDataset(dataset['validation'], transform=transform_val)
    test_dataset = ChestXRayDataset(dataset['test'], transform=transform_val)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

    print("✅ DataLoaders created successfully!")
    return train_loader, val_loader, test_loader


# ======================================
# This block runs only when file is run directly
# ======================================
if __name__ == "__main__":
    dataset = load_dataset("hf-vision/chest-xray-pneumonia")
    
    print("Plotting samples...")
    plot_samples(dataset['train'])

    print("Plotting class distributions...")
    for split_name, split_data in dataset.items():
        plot_class_distribution(split_data, split_name.capitalize())

    print("Checking image channels...")
    check_image_channels(dataset['train'])

    print("Plotting pixel intensity histograms...")
    plot_pixel_intensity_histograms(dataset['train'])

    print("Plotting single image histogram...")
    plot_single_image_histogram(0, dataset['train'])