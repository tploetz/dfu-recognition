from utils import *
from filetype_convert import *
import os
import cv2
import shutil
import numpy as np
import argparse
from sklearn.model_selection import train_test_split

def split_data_into_train_test(data_dir, test_ratio=0.2, random_state=42):
    """
    Split data into train and test sets
    """
    # Get all image files
    img_dir = os.path.join(data_dir, 'images')
    label_dir = os.path.join(data_dir, 'labels')
    
    img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    if len(img_files) == 0:
        print(f"No images found in {data_dir}")
        return
    
    # Split into train and test
    train_files, test_files = train_test_split(
        img_files, test_size=test_ratio, random_state=random_state, shuffle=True
    )
    
    print(f"Total images: {len(img_files)}")
    print(f"Train images: {len(train_files)}")
    print(f"Test images: {len(test_files)}")
    
    # Create train and test directories
    train_img_dir = os.path.join(data_dir, 'train', 'images')
    train_label_dir = os.path.join(data_dir, 'train', 'labels')
    test_img_dir = os.path.join(data_dir, 'test', 'images')
    test_label_dir = os.path.join(data_dir, 'test', 'labels')
    
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)
    os.makedirs(test_label_dir, exist_ok=True)
    
    # Move train files
    print("Moving train files...")
    for img_file in train_files:
        base_name = os.path.splitext(img_file)[0]
        img_src = os.path.join(img_dir, img_file)
        label_src = os.path.join(label_dir, base_name + '.png')
        
        img_dst = os.path.join(train_img_dir, img_file)
        label_dst = os.path.join(train_label_dir, base_name + '.png')
        
        if os.path.exists(img_src):
            shutil.move(img_src, img_dst)
        if os.path.exists(label_src):
            shutil.move(label_src, label_dst)
    
    # Move test files
    print("Moving test files...")
    for img_file in test_files:
        base_name = os.path.splitext(img_file)[0]
        img_src = os.path.join(img_dir, img_file)
        label_src = os.path.join(label_dir, base_name + '.png')
        
        img_dst = os.path.join(test_img_dir, img_file)
        label_dst = os.path.join(test_label_dir, base_name + '.png')
        
        if os.path.exists(img_src):
            shutil.move(img_src, img_dst)
        if os.path.exists(label_src):
            shutil.move(label_src, label_dst)
    
    # Remove empty original directories
    try:
        os.rmdir(img_dir)
        os.rmdir(label_dir)
    except OSError:
        pass
    
    print(f"Train/test split completed for {data_dir}")

def split_data(input_dir, output_dir, split_train_test=True, test_ratio=0.2):
    """
    Split data into normal and ulcer categories based on mask content
    """
    # Input directories
    resized_img_dir = os.path.join(input_dir, 'images')
    resized_mask_dir = os.path.join(input_dir, 'labels')
    
    # Check if input directories exist
    if not os.path.exists(resized_img_dir):
        print(f"Error: Images directory not found: {resized_img_dir}")
        return 0, 0
    
    if not os.path.exists(resized_mask_dir):
        print(f"Error: Labels directory not found: {resized_mask_dir}")
        return 0, 0
    
    # Output directories
    ducss_ulcer_dir = os.path.join(output_dir, 'ulcer')
    ducss_ulcer_img_dir = os.path.join(ducss_ulcer_dir, 'images')
    ducss_ulcer_label_dir = os.path.join(ducss_ulcer_dir, 'labels')
    os.makedirs(ducss_ulcer_img_dir, exist_ok=True)
    os.makedirs(ducss_ulcer_label_dir, exist_ok=True)
    
    ducss_normal_dir = os.path.join(output_dir, 'normal')
    ducss_normal_img_dir = os.path.join(ducss_normal_dir, 'images')
    ducss_normal_label_dir = os.path.join(ducss_normal_dir, 'labels')
    os.makedirs(ducss_normal_img_dir, exist_ok=True)
    os.makedirs(ducss_normal_label_dir, exist_ok=True)
    
    # Get all image files
    img_files = [f for f in os.listdir(resized_img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    img_num = len(img_files)
    
    if img_num == 0:
        print("No image files found in the input directory")
        return 0, 0
    
    print(f"Found {img_num} images to process")
    
    ulcer_count = 0
    normal_count = 0
    processed_count = 0
    
    # Split data based on mask content
    for img_name in img_files:
        processed_count += 1
        printProgressBar(processed_count, img_num, prefix='Splitting:', suffix='Complete')
        
        img_path = os.path.join(resized_img_dir, img_name)
        base_name = os.path.splitext(img_name)[0]
        mask_path = os.path.join(resized_mask_dir, base_name + '.png')
        
        if not os.path.exists(mask_path):
            print(f"\nWarning: No matching mask found for {img_name}")
            continue
        
        try:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"\nError: Cannot read mask {mask_path}")
                continue
            
            # Check if mask has any non-zero pixels (ulcer present)
            if np.sum(mask) > 0:
                # Copy to ulcer directory
                shutil.copy(img_path, os.path.join(ducss_ulcer_img_dir, img_name))
                shutil.copy(mask_path, os.path.join(ducss_ulcer_label_dir, base_name + '.png'))
                ulcer_count += 1
            else:
                # Copy to normal directory
                shutil.copy(img_path, os.path.join(ducss_normal_img_dir, img_name))
                shutil.copy(mask_path, os.path.join(ducss_normal_label_dir, base_name + '.png'))
                normal_count += 1
                
        except Exception as e:
            print(f"\nError processing {img_name}: {str(e)}")
            continue
    
    print(f"\nData splitting completed!")
    print(f"Ulcer images: {ulcer_count}")
    print(f"Normal images: {normal_count}")
    print(f"Total processed: {ulcer_count + normal_count}")
    
    # Convert all files to PNG format
    print(f"\nConverting all files to PNG format...")
    convert_to_png(ducss_ulcer_img_dir, ducss_ulcer_label_dir)
    convert_to_png(ducss_normal_img_dir, ducss_normal_label_dir)
    
    # Split into train and test sets if requested
    if split_train_test:
        print(f"\n=== Starting train/test split ===")
        if ulcer_count > 0:
            print(f"Splitting ulcer data...")
            split_data_into_train_test(ducss_ulcer_dir, test_ratio, 42)
        
        if normal_count > 0:
            print(f"Splitting normal data...")
            split_data_into_train_test(ducss_normal_dir, test_ratio, 42)
        
        print(f"Train/test split completed!")
    
    return ulcer_count, normal_count

def convert_to_png(img_dir, label_dir):
    """
    Convert all image and label files to PNG format
    """
    print("Converting images to PNG...")
    jpg2png(img_dir)
    
    print("Converting labels to PNG...")
    jpg2png(label_dir)
    
    print("File format conversion completed!")

def main():
    parser = argparse.ArgumentParser(description='Split data into normal/ulcer categories and optionally into train/test sets')
    parser.add_argument('--input_dir', '-i', type=str, default='./data/DFU_resized',
                        help='Input directory containing resized images and labels')
    parser.add_argument('--output_dir', '-o', type=str, default='./data/DFU_split',
                        help='Output directory for split data')
    parser.add_argument('--no_train_test_split', action='store_true',
                        help='Skip train/test split (default: perform train/test split)')
    parser.add_argument('--test_ratio', '-r', type=float, default=0.2,
                        help='Ratio of test set (default: 0.2)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for reproducible splits (default: 42)')
    
    args = parser.parse_args()
    
    split_data(args.input_dir, args.output_dir, 
               split_train_test=not args.no_train_test_split, 
               test_ratio=args.test_ratio)

if __name__ == "__main__":
    main()