import os
import pandas as pd

# Paths
csv_file = "/content/drive/MyDrive/FYP/crazyDataset/valid_annotations.csv"
labels_folder = "/content/drive/MyDrive/FYP/nlabelsValid"

os.makedirs(labels_folder, exist_ok=True)

# Load CSV
df = pd.read_csv(csv_file)

class_mapping = {"bus": 0, "car": 1, "jeepney": 2, "motorcycle": 3, "tricycle": 4, "truck": 5, "van": 6 }  # Adjust as needed

# Process each image
for image_name in df["filename"].unique():
    annotations = df[df["filename"] == image_name]

    # Open corresponding YOLO label file
    label_file_path = os.path.join(labels_folder, image_name.replace(".jpg", ".txt"))
    with open(label_file_path, "w") as f:
        for _, row in annotations.iterrows():
            class_id = class_mapping[row["class"]]
            img_w, img_h = row["width"], row["height"]  # Use provided dimensions
            x_min, y_min, x_max, y_max = row["xmin"], row["ymin"], row["xmax"], row["ymax"]

            # Convert to YOLO format (normalized values)
            x_center = (x_min + x_max) / 2 / img_w
            y_center = (y_min + y_max) / 2 / img_h
            width = (x_max - x_min) / img_w
            height = (y_max - y_min) / img_h

            # Write annotation in YOLO format
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

print("✅ Conversion complete! YOLO labels saved in:", labels_folder)

import shutil
import os

def move_files(source_folder, destination_folder):

    if not os.path.exists(source_folder):
        print(f"Error: Source folder '{source_folder}' does not exist.")
        return

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        print(f"Created destination folder '{destination_folder}'")

    for filename in os.listdir(source_folder):
        source_path = os.path.join(source_folder, filename)
        destination_path = os.path.join(destination_folder, filename)

        if os.path.isfile(source_path):
            try:
                shutil.move(source_path, destination_folder)
                print(f"Moved '{filename}' to '{destination_folder}'")
            except Exception as e:
                print(f"Error moving '{filename}': {e}")
        else:
            print(f"Skipping '{filename}' (not a file)")

source_folder = "/content/drive/MyDrive/FYP/nhelmetvalid/labels"
destination_folder = "/content/drive/MyDrive/FYP/crazyDataset/dataset2/valid/labels"
move_files(source_folder, destination_folder)

import os
import shutil

# Define paths
labels_folder = "/content/drive/MyDrive/FYP/aio/valid/labels"
images_folder = "/content/drive/MyDrive/FYP/aio/valid/images"
output_labels_folder = "/content/drive/MyDrive/FYP/nhelmetvalid/labels"
output_images_folder = "/content/drive/MyDrive/FYP/nhelmetvalid/images"

# Ensure output directories exist
os.makedirs(output_labels_folder, exist_ok=True)
os.makedirs(output_images_folder, exist_ok=True)

# Class mapping
class_mapping = {4: 7, 5: 8}

# Process each label file
for label_file in os.listdir(labels_folder):
    if not label_file.endswith(".txt"):
        continue  # Skip non-label files

    input_label_path = os.path.join(labels_folder, label_file)
    output_label_path = os.path.join(output_labels_folder, label_file)

    # Read and filter label file
    new_lines = []
    with open(input_label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])

            if class_id in class_mapping:
                # Modify class label
                parts[0] = str(class_mapping[class_id])
                new_lines.append(" ".join(parts))

    if new_lines:
        with open(output_label_path, "w") as f:
            f.write("\n".join(new_lines) + "\n")

        image_name = label_file.replace(".txt", ".jpg")
        input_image_path = os.path.join(images_folder, image_name)
        output_image_path = os.path.join(output_images_folder, image_name)

        if os.path.exists(input_image_path):
            shutil.copy(input_image_path, output_image_path)

print("✅ Extraction and label modification complete!")

import os
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import shutil

def augment_helmet_data(base_dir, output_dir):

    image_dir = os.path.join(base_dir, 'images')
    label_dir = os.path.join(base_dir, 'labels')
    output_image_dir = os.path.join(output_dir, 'images')
    output_label_dir = os.path.join(output_dir, 'labels')

    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    # Define augmentation pipeline
    augmentations = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.CoarseDropout(max_holes=4, max_height=30, max_width=30, min_holes=1, fill_value=0, p=0.5),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    # Process each image and corresponding label
    for filename in tqdm(os.listdir(image_dir)):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            base_name = os.path.splitext(filename)[0]
            image_path = os.path.join(image_dir, filename)
            label_path = os.path.join(label_dir, base_name + '.txt')

            # Check if the label exists
            if not os.path.exists(label_path):
                continue

            # Read image and check if loaded
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Failed to load {image_path}")
                continue

            with open(label_path, 'r') as f:
                lines = f.readlines()

            # Parse YOLO labels
            bboxes = []
            class_labels = []
            for line in lines:
                cls, x, y, w, h = map(float, line.strip().split())
                if int(cls) in [5, 6, 7, 8]:  # Helmet-related classes
                    bboxes.append([x, y, w, h])
                    class_labels.append(int(cls))

            if not bboxes:
                continue

            # Apply augmentation
            augmented = augmentations(image=image, bboxes=bboxes, class_labels=class_labels)
            aug_image = augmented['image']
            aug_bboxes = augmented['bboxes']
            aug_class_labels = augmented['class_labels']

            # Save augmented image
            aug_image_path = os.path.join(output_image_dir, f'aug_{filename}')
            cv2.imwrite(aug_image_path, aug_image)

            # Save augmented labels
            aug_label_path = os.path.join(output_label_dir, f'aug_{base_name}.txt')
            with open(aug_label_path, 'w') as f:
                for cls, (x, y, w, h) in zip(aug_class_labels, aug_bboxes):
                    f.write(f"{cls} {x} {y} {w} {h}\n")

# Define paths
base_dir = "/content/drive/MyDrive/FYP/Cars Detection/train"
output_dir = "/content/drive/MyDrive/FYP/Cars Detection"

# Augment helmet-related data
augment_helmet_data(base_dir, output_dir)

"""# MODEL TRAINING"""

!pip install ultralytics --upgrade

import torch
import torch.nn as nn
from ultralytics import YOLO

class LSKA(nn.Module):
    def __init__(self, in_channels):
        super(LSKA, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 9), padding=(0, 4), groups=in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=(9, 1), padding=(4, 0), groups=in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.conv1(x)
        attention = self.conv2(attention)
        return x * self.sigmoid(attention)

class SPPF_LSKA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super(SPPF_LSKA, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, kernel_size // 2)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, kernel_size // 2)
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, kernel_size // 2)
        self.lska = LSKA(out_channels)
        self.conv5 = nn.Conv2d(out_channels * 4, out_channels, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.conv2(x)
        y2 = self.conv3(y1)
        y3 = self.conv4(y2)
        x = torch.cat([x, y1, y2, y3], 1)
        x = self.conv5(x)
        return self.lska(x)

class STE_Neck(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(STE_Neck, self).__init__()
        self.upsample1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = nn.Conv2d(in_channels_list[0], out_channels, 1, 1)
        self.conv2 = nn.Conv2d(in_channels_list[1], out_channels, 1, 1)
        self.conv3 = nn.Conv2d(out_channels * 2, out_channels, 3, 1, 1)
        self.conv4 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

    def forward(self, x1, x2):
        x1 = self.upsample1(x1)
        x2 = self.conv1(x2)
        x1 = self.conv2(x1)
        x = torch.cat([x1, x2], 1)
        x = self.conv3(x)
        return self.conv4(x)

from collections import Counter
import os

# Paths to the labels folder
labels_path = '/content/drive/MyDrive/FYP/crazyDataset/dataset2/train/labels'

# Counter for class distribution
class_distribution = Counter()

# Iterate through all label files
for label_file in os.listdir(labels_path):
    if label_file.endswith('.txt'):  # Process only text files
        with open(os.path.join(labels_path, label_file), 'r') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    class_id = line.split()[0]  # Extract the class ID (first element)
                    class_distribution[class_id] += 1

# Print the class distribution
print("Class Distribution:", class_distribution)

class_counts = torch.tensor([97, 14719, 123, 658, 1, 213, 781, 286, 818])
total_samples = class_counts.sum()
class_weights = total_samples / class_counts
class_weights = class_weights / class_weights.sum()

class CustomYOLO(YOLO):
    def compute_loss(self, preds, targets):
        base_loss, components = super().compute_loss(preds, targets)

        # Weighted classification loss
        classification_loss = components[0]
        class_ids = targets[:, 1].long()  # Assuming target format [image_idx, class_id, x_center, y_center, width, height]
        weights = class_weights[class_ids].to(classification_loss.device)
        weighted_classification_loss = (classification_loss * weights).mean()

        components[0] = weighted_classification_loss
        total_loss = sum(components)
        return total_loss, components

checkpoint_path = "yolov8n.pt"
pretrained_model = YOLO(checkpoint_path)

# Replace Pretrained YOLO Model with Custom Model
custom_model = CustomYOLO(checkpoint_path)

custom_model.train(
    data="/content/drive/MyDrive/FYP/crazyDataset/dataset2/data2.yaml",
    epochs=100,
    batch=64,
    lr0=1e-5,
    project="/content/drive/MyDrive/FYP/crazyDataset/dataset2/finetuned_model",
    name="custom_yolo_finetuned",
    device=0
)