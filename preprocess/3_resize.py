
from utils import *
import os
import cv2
import argparse

def main():
    parser = argparse.ArgumentParser(description='Resize images and masks')
    parser.add_argument('--input_dir', '-i', type=str, default='./data/DFU',
                        help='Input directory containing images and labels subdirectories')
    parser.add_argument('--output_dir', '-o', type=str, default='./data/DFU_resized',
                        help='Output directory for resized images and labels')
    parser.add_argument('--size', '-s', type=int, nargs=2, default=[512, 512],
                        help='Target size for resizing (width height), default: 512 512')
    
    args = parser.parse_args()
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    target_size = tuple(args.size)
    
    # Auto-match images and labels directories
    ducss_img_dir = os.path.join(input_dir, 'images')
    ducss_label_dir = os.path.join(input_dir, 'labels')
    
    # Check if input directories exist
    if not os.path.exists(ducss_img_dir):
        print(f"Error: Images directory not found: {ducss_img_dir}")
        return
    
    if not os.path.exists(ducss_label_dir):
        print(f"Error: Labels directory not found: {ducss_label_dir}")
        return
    
    # Create output directories
    resized_img_dir = os.path.join(output_dir, 'images')
    resized_mask_dir = os.path.join(output_dir, 'labels')
    os.makedirs(resized_img_dir, exist_ok=True)
    os.makedirs(resized_mask_dir, exist_ok=True)
    
    # Get all image files
    img_files = [f for f in os.listdir(ducss_img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    img_num = len(img_files)
    
    if img_num == 0:
        print("No image files found in the input directory")
        return
    
    print(f"Found {img_num} images to process")
    print(f"Target size: {target_size}")
    
    img_count = 0
    success_count = 0
    
    for img_name in img_files:
        img_count += 1
        printProgressBar(img_count, img_num, prefix='Progress:', suffix='Complete')
        
        img_path = os.path.join(ducss_img_dir, img_name)
        
        # Auto-match corresponding label files
        base_name = os.path.splitext(img_name)[0]
        possible_mask_extensions = ['.png', '.jpg', '.jpeg']
        mask_path = None
        
        for ext in possible_mask_extensions:
            potential_mask_path = os.path.join(ducss_label_dir, base_name + ext)
            if os.path.exists(potential_mask_path):
                mask_path = potential_mask_path
                break
        
        if mask_path is None:
            print(f"\nWarning: No matching mask found for {img_name}")
            continue
        
        try:
            # Read image and mask
            image = cv2.imread(img_path)
            mask = cv2.imread(mask_path)
            
            if image is None:
                print(f"\nError: Cannot read image {img_name}")
                continue
                
            if mask is None:
                print(f"\nError: Cannot read mask {mask_path}")
                continue
            
            # Resize images and masks
            resized_img = resize_and_pad(image, target_size)
            resized_mask = resize_and_pad(mask, target_size)
            
            # Save resized image and mask
            resized_img_path = os.path.join(resized_img_dir, img_name)
            resized_mask_name = base_name + '.png'  # Save masks as PNG format
            resized_mask_path = os.path.join(resized_mask_dir, resized_mask_name)
            
            cv2.imwrite(resized_img_path, resized_img)
            cv2.imwrite(resized_mask_path, resized_mask)
            
            success_count += 1
            
        except Exception as e:
            print(f"\nError processing {img_name}: {str(e)}")
            continue
    
    print(f"\nProcessing completed!")
    print(f"Successfully processed: {success_count}/{img_num} images")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main()