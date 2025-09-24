#!/usr/bin/env python3
"""
Generate YOLO format label files
Generate bounding boxes from mask images and merge ulcer and normal datasets
"""

import os
import cv2
import numpy as np
import shutil
from pathlib import Path
import yaml
from tqdm import tqdm


def mask_to_bbox(mask, threshold=0):
    """
    Generate bounding boxes from mask
    
    Args:
        mask: Binary mask image
        threshold: Threshold value, pixels greater than this are considered foreground
    
    Returns:
        list: List of bounding boxes in format [x_center, y_center, width, height] (normalized coordinates)
    """
    # Binarize mask
    binary_mask = (mask > threshold).astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bboxes = []
    h, w = mask.shape
    
    for contour in contours:
        # Calculate bounding box
        x, y, width, height = cv2.boundingRect(contour)
        
        # Filter out small regions
        if width < 10 or height < 10:
            continue
            
        # Convert to YOLO format (normalized coordinates)
        x_center = (x + width / 2) / w
        y_center = (y + height / 2) / h
        norm_width = width / w
        norm_height = height / h
        
        bboxes.append([x_center, y_center, norm_width, norm_height])
    
    return bboxes


def process_dataset(source_dir, target_dir, class_id, split_name):
    """
    Process single dataset (ulcer or normal)
    
    Args:
        source_dir: Source data directory
        target_dir: Target data directory
        class_id: Class ID (0: ulcer, normal has no labels)
        split_name: Dataset split name (train/test)
    """
    images_dir = os.path.join(source_dir, split_name, 'images')
    labels_dir = os.path.join(source_dir, split_name, 'labels')
    
    # Keep original split names
    target_images_dir = os.path.join(target_dir, split_name, 'images')
    target_labels_dir = os.path.join(target_dir, split_name, 'labels')
    
    # Create target directories
    os.makedirs(target_images_dir, exist_ok=True)
    os.makedirs(target_labels_dir, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Processing {split_name} dataset: {len(image_files)} files")
    
    for image_file in tqdm(image_files, desc=f"Processing {split_name}"):
        # Build file paths
        image_path = os.path.join(images_dir, image_file)
        label_path = os.path.join(labels_dir, image_file)
        
        # Target file paths
        target_image_path = os.path.join(target_images_dir, image_file)
        target_label_path = os.path.join(target_labels_dir, os.path.splitext(image_file)[0] + '.txt')
        
        # Copy image file
        shutil.copy2(image_path, target_image_path)
        
        # Process label file
        if os.path.exists(label_path):
            # Read mask
            mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            
            if class_id == 0:  # ulcer class, has lesion regions
                # Generate bounding boxes from mask
                bboxes = mask_to_bbox(mask)
                
                # Write YOLO format labels
                with open(target_label_path, 'w') as f:
                    for bbox in bboxes:
                        f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
            else:  # normal class, no lesion regions, don't create label file
                # For normal class, don't create label file (indicates no ulcer)
                pass
        else:
            # If no corresponding label file, don't create label file
            pass


def merge_datasets():
    """
    Merge ulcer and normal datasets to generate YOLO format dataset
    """
    # Define paths
    base_dir = "/coc/pcba1/yshi457/DUCSS-DFU-Recognition/data"
    source_dir = os.path.join(base_dir, "DFU_split")
    target_dir = os.path.join(base_dir, "yolo_dataset")
    
    # Create target directory structure
    os.makedirs(os.path.join(target_dir, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'test', 'images'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'train', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'test', 'labels'), exist_ok=True)
    
    print("Starting dataset processing...")
    
    # Process normal dataset (no label files)
    normal_dir = os.path.join(source_dir, "normal")
    if os.path.exists(normal_dir):
        print("\nProcessing normal dataset...")
        process_dataset(normal_dir, target_dir, class_id=1, split_name='train')  # class_id=1 means normal, no labels
        process_dataset(normal_dir, target_dir, class_id=1, split_name='test')
    
    # Process ulcer dataset (class_id = 0)
    ulcer_dir = os.path.join(source_dir, "ulcer")
    if os.path.exists(ulcer_dir):
        print("\nProcessing ulcer dataset...")
        process_dataset(ulcer_dir, target_dir, class_id=0, split_name='train')  # class_id=0 means ulcer
        process_dataset(ulcer_dir, target_dir, class_id=0, split_name='test')
    
    print(f"\nDataset processing completed! Output directory: {target_dir}")
    
    # Statistics
    for split in ['train', 'test']:
        images_count = len([f for f in os.listdir(os.path.join(target_dir, split, 'images')) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        labels_count = len([f for f in os.listdir(os.path.join(target_dir, split, 'labels')) 
                           if f.lower().endswith('.txt')])
        print(f"{split}: {images_count} images, {labels_count} label files")
    
    return target_dir


def generate_yaml_config(dataset_dir):
    """
    Generate YOLO format yaml configuration file
    
    Args:
        dataset_dir: Dataset directory
    """
    yaml_path = os.path.join(dataset_dir, 'dataset.yaml')
    
    # Write YAML file in the desired format
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write("# YOLO\n")
        f.write(f"path: {dataset_dir}\n")
        f.write("train: train/images\n")
        f.write("val: test/images\n")
        f.write("test: test/images\n")
        f.write("nc: 1\n")
        f.write("names: ['ulcer']\n")
        f.write("\n")
    
    print(f"YAML configuration file generated: {yaml_path}")
    print("Configuration content:")
    print(f"  - Number of classes: 1")
    print(f"  - Class names: ['ulcer']")
    print(f"  - Training set: train/images")
    print(f"  - Validation set: test/images")
    print(f"  - Test set: test/images")


def main():
    """
    Main function
    """
    print("=" * 60)
    print("DFU Dataset YOLO Format Conversion Tool")
    print("=" * 60)
    
    # Merge datasets
    dataset_dir = merge_datasets()
    
    # Generate YAML configuration file
    generate_yaml_config(dataset_dir)
    
    print("\n" + "=" * 60)
    print("Processing completed!")
    print("=" * 60)
    print(f"Dataset location: {dataset_dir}")
    print("Directory structure:")
    print("├── train/")
    print("│   ├── images/")
    print("│   └── labels/")
    print("├── test/")
    print("│   ├── images/")
    print("│   └── labels/")
    print("└── dataset.yaml")


if __name__ == "__main__":
    main()