
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Augmentation Script - Perform data augmentation on DFU dataset
Uses Augmentor library to augment ulcer images and corresponding masks
"""

import os
import cv2
import shutil
import argparse
import Augmentor
from utils import printProgressBar
from PIL import Image

def setup_augmentation_pipeline(image_dir, label_dir):
    """
    Setup data augmentation pipeline
    
    Args:
        image_dir: Image directory path
        label_dir: Label directory path
    
    Returns:
        Augmentor.Pipeline: Configured augmentation pipeline
    """
    p = Augmentor.Pipeline(image_dir)
    p.ground_truth(label_dir)
    
    # Configure various augmentation operations
    p.flip_random(probability=0.8)  # Random flip
    p.rotate90(probability=0.8)     # 90-degree rotation
    p.rotate(probability=0.8, max_left_rotation=25, max_right_rotation=25)  # Random rotation
    p.zoom_random(probability=1, percentage_area=0.8)  # Random zoom
    p.random_distortion(probability=0.5, grid_width=10, grid_height=10, magnitude=20)  # Random distortion
    p.shear(probability=0.8, max_shear_left=10, max_shear_right=10)  # Shear transformation
    
    return p

def process_augmented_images(image_dir, label_dir, target_count, batch_size=1000):
    """
    Process augmented images and filter out images without masks
    
    Args:
        image_dir: Original image directory
        label_dir: Original label directory
        target_count: Target number of images
        batch_size: Number of images to process per batch
    
    Returns:
        tuple: (total image count, no mask image count)
    """
    all_count = len(os.listdir(image_dir))
    no_mask = 0
    aug_out_dir = os.path.join(image_dir, 'output')
    
    print(f"Starting data augmentation, target count: {target_count}")
    print(f"Current image count: {all_count}")
    
    while all_count < target_count:
        print(f"Current progress: {all_count}/{target_count}")
        
        # Generate a batch of augmented images
        p = setup_augmentation_pipeline(image_dir, label_dir)
        p.sample(batch_size)
        
        # Process generated images
        if os.path.exists(aug_out_dir):
            for filename in os.listdir(aug_out_dir):
                if filename.startswith('images'):
                    all_count += 1
                    
                    img_path = os.path.join(aug_out_dir, filename)
                    mask_name = "_groundtruth_(1)_" + filename.replace('original_', '')
                    mask_path = os.path.join(aug_out_dir, mask_name)
                    
                    # Check if mask file exists
                    if not os.path.exists(mask_path):
                        print(f"Warning: Mask file does not exist: {mask_name}")
                        all_count -= 1
                        if os.path.exists(img_path):
                            os.remove(img_path)
                        continue
                    
                    try:
                        # Read and process mask
                        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        if mask is None:
                            print(f"Warning: Cannot read mask file: {mask_path}")
                            all_count -= 1
                            os.remove(img_path)
                            os.remove(mask_path)
                            continue
                            
                        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
                        
                        # Check if mask is empty
                        notBlank = mask.any()
                        if not notBlank:
                            no_mask += 1
                            all_count -= 1
                            os.remove(img_path)
                            os.remove(mask_path)
                            continue
                        
                        # Save augmented image and mask
                        new_mask_path = os.path.join(label_dir, filename.replace('images_original_', ''))
                        new_img_path = os.path.join(image_dir, filename.replace('images_original_', ''))
                        
                        # Ensure filenames end with .png
                        if not new_img_path.endswith('.png'):
                            new_img_path = new_img_path.replace('.jpg', '.png')
                        if not new_mask_path.endswith('.png'):
                            new_mask_path = new_mask_path.replace('.jpg', '.png')
                        
                        # Convert image format to PNG
                        if filename.endswith('.jpg'):
                            with Image.open(img_path) as img:
                                img.save(new_img_path)
                        else:
                            shutil.copy(img_path, new_img_path)
                        
                        # Save mask
                        cv2.imwrite(new_mask_path, mask)
                        
                        # Clean up temporary files
                        os.remove(img_path)
                        os.remove(mask_path)
                        
                    except Exception as e:
                        print(f"Error processing image {filename}: {str(e)}")
                        all_count -= 1
                        if os.path.exists(img_path):
                            os.remove(img_path)
                        if os.path.exists(mask_path):
                            os.remove(mask_path)
                        continue
        
        print(f"Current total: {all_count} images, {no_mask} images without mask")
    
    os.rmdir(aug_out_dir)
    return all_count, no_mask

def augment_data(data_dir, augmentation_factor=5):
    """
    Perform data augmentation on the dataset
    
    Args:
        data_dir: Data root directory
        augmentation_factor: Augmentation factor
    """
    # Set paths
    # ducss_ulcer_dir = os.path.join(data_dir, 'ulcer')
    original_img_dir = os.path.join(data_dir, 'images')
    original_label_dir = os.path.join(data_dir, 'labels')
    
    # Check if directories exist
    if not os.path.exists(original_img_dir):
        print(f"Error: Image directory does not exist: {original_img_dir}")
        return
    
    if not os.path.exists(original_label_dir):
        print(f"Error: Label directory does not exist: {original_label_dir}")
        return
    
    # Calculate target count
    original_count = len(os.listdir(original_img_dir))
    target_count = original_count * augmentation_factor
    
    print(f"=== Starting Data Augmentation ===")
    print(f"Original image count: {original_count}")
    print(f"Target image count: {target_count}")
    print(f"Augmentation factor: {augmentation_factor}")
    
    # Execute data augmentation
    final_count, no_mask_count = process_augmented_images(
        original_img_dir, 
        original_label_dir, 
        target_count
    )
    
    print(f"=== Data Augmentation Completed ===")
    print(f"Final image count: {final_count}")
    print(f"No mask image count: {no_mask_count}")
    print(f"Effective augmented images: {final_count - original_count}")

def main():
    parser = argparse.ArgumentParser(description='DFU Data Augmentation Script')
    parser.add_argument('--data_dir', '-d', type=str, default='./data/DFU_split/ulcer/train',
                        help='Data root directory path (default: ./data/DFU_split/ulcer/train)')
    parser.add_argument('--augmentation_factor', '-f', type=int, default=5,
                        help='Data augmentation factor (default: 5)')
    parser.add_argument('--batch_size', '-b', type=int, default=1000,
                        help='Number of images to process per batch (default: 1000)')
    
    args = parser.parse_args()
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory does not exist: {args.data_dir}")
        return
    
    # Execute data augmentation
    augment_data(args.data_dir, args.augmentation_factor)

if __name__ == "__main__":
    main()