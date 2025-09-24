import argparse
import os
import torch
import imageio
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from typing import List, Union
from tqdm import tqdm
from PIL import Image
from sam2_unet.SAM2UNet import SAM2UNet
from sam2_unet.dataset import TestDataset


parser = argparse.ArgumentParser(description='SAM2-UNet Detection Tool - Supports both folder and single file calling methods')
parser.add_argument("--checkpoint", type=str, required=True,
                help="path to the checkpoint of sam2-unet")
parser.add_argument("--input", "-i", type=str, required=False, default="data/DFU_split/ulcer/test/images",
                    help="input path (folder or single image file)")
parser.add_argument("--output", "-o", type=str, required=False, default="prediction/dfu",
                    help="path to save the predicted masks")
parser.add_argument("--cuda_idx", default=0, type=int,
                    help="CUDA device index")
parser.add_argument("--image_size", type=int, default=352,
                    help="input image size")
parser.add_argument("--batch_size", type=int, default=1,
                    help="batch size for processing")
parser.add_argument("--use_dataset", action="store_true", default=True,
                    help="use original TestDataset for folder detection (default: True)")
args = parser.parse_args()

# Device configuration
if torch.cuda.is_available():
    device = torch.device(f'cuda:{args.cuda_idx}')
    print(f"Using GPU device: {device}")
    print(f"Available GPU count: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
else:
    device = torch.device('cpu')
    print("CUDA not available, using CPU")


def get_image_files(folder_path: str) -> List[str]:
    """Get all image files in the folder"""
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    image_files = []
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in supported_formats):
                image_files.append(os.path.join(root, file))
    
    return sorted(image_files)


def preprocess_image(image_path: str, target_size: int = 352):
    """Preprocess single image"""
    try:
        # Read image
        image = imageio.imread(image_path)
        original_shape = image.shape[:2]  # Save original size
        
        # Convert to RGB format
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = image
        elif len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        else:
            raise ValueError(f"Unsupported image format: {image.shape}")
        
        # Resize
        pil_image = Image.fromarray(image)
        pil_image = pil_image.resize((target_size, target_size), Image.LANCZOS)
        image = np.array(pil_image)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor, original_shape  # Return actual original size
        
    except Exception as e:
        print(f"Failed to preprocess image {image_path}: {e}")
        return None, None


def predict_single_image(model, image_tensor, device, original_shape):
    """Predict single image"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        res, _, _ = model(image_tensor)
        res = torch.sigmoid(res)
        
        # Resize output back to original size
        res = F.upsample(res, size=original_shape, mode='bilinear', align_corners=False)
        
        res = res.data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        res = (res * 255).astype(np.uint8)
        return res


def detect_single_image(model, image_path: str, output_dir: str, device, image_size: int = 352):
    """Detect single image"""
    print(f"Detecting image: {os.path.basename(image_path)}")
    
    # Preprocess image
    image_tensor, original_shape = preprocess_image(image_path, image_size)
    if image_tensor is None:
        return False
    
    
    # Prediction - always maintain original size
    prediction = predict_single_image(model, image_tensor, device, original_shape)
    
    
    # Save result
    output_filename = Path(image_path).stem + "_sam_mask.png"
    output_path = os.path.join(output_dir, output_filename)
    imageio.imsave(output_path, prediction)
    
    print(f"✅ Result saved: {output_path}")
    return True


def detect_folder_with_dataset(model, folder_path: str, output_dir: str, device, image_size: int = 352):
    """Use original TestDataset approach for folder detection"""
    print(f"Using TestDataset approach for folder detection: {folder_path}")
    
    # Create a simplified TestDataset without ground truth
    class SimpleTestDataset:
        def __init__(self, image_root, size):
            self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            self.images = sorted(self.images)
            
            # Use the same preprocessing as the original TestDataset
            from torchvision import transforms
            self.transform = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
            self.size = len(self.images)
            self.index = 0
        
        def rgb_loader(self, path):
            with open(path, 'rb') as f:
                img = Image.open(f)
                return img.convert('RGB')
        
        def load_data(self):
            if self.index >= self.size:
                self.index = 0
            image = self.rgb_loader(self.images[self.index])
            original_shape = image.size[::-1]  # PIL returns (width, height), we need (height, width)
            image = self.transform(image).unsqueeze(0)
            name = os.path.basename(self.images[self.index])
            self.index += 1
            return image, original_shape, name
    
    try:
        test_loader = SimpleTestDataset(folder_path, image_size)
        print(f"SimpleTestDataset loaded successfully, number of images: {test_loader.size}")
    except Exception as e:
        print(f"❌ SimpleTestDataset loading failed: {e}")
        print("Falling back to new folder detection method...")
        return detect_folder_new(model, folder_path, output_dir, device, image_size)
    
    # Use progress bar for batch processing
    with tqdm(total=test_loader.size, desc="SAM2-UNet detection progress", unit="images") as pbar:
        success_count = 0
        for i in range(test_loader.size):
            try:
                with torch.no_grad():
                    image, gt_shape, name = test_loader.load_data()
                    
                    # Update progress bar description
                    pbar.set_description(f"Detecting: {name}")
                    
                    # Process image
                    image = image.to(device)
                    
                    # Model prediction - completely based on original implementation
                    res, _, _ = model(image)
                    res = torch.sigmoid(res)
                    res = F.upsample(res, size=gt_shape, mode='bilinear', align_corners=False)
                    res = res.sigmoid().data.cpu()
                    res = res.numpy().squeeze()
                    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                    res = (res * 255).astype(np.uint8)
                    
                    # Save result
                    output_filename = name[:-4] + "_sam_mask.png"
                    output_path = os.path.join(output_dir, output_filename)
                    imageio.imsave(output_path, res)
                    
                    success_count += 1
                    pbar.set_postfix({"Success": success_count})
                    
            except Exception as e:
                print(f"❌ Error processing image {i}: {e}")
                pbar.set_postfix({"Failed": str(e)})
            
            pbar.update(1)
    
    print(f"\n✅ Folder detection completed! Successfully processed {success_count}/{test_loader.size} images")


def detect_folder_new(model, folder_path: str, output_dir: str, device, image_size: int = 352):
    """New folder detection method (process images one by one)"""
    # Get all image files
    image_files = get_image_files(folder_path)
    
    if not image_files:
        print(f"No supported image files found in folder {folder_path}")
        return
    
    print(f"Found {len(image_files)} image files")
    
    # Use progress bar for batch processing
    with tqdm(total=len(image_files), desc="SAM2-UNet detection progress", unit="images") as pbar:
        success_count = 0
        for image_path in image_files:
            pbar.set_description(f"Detecting: {os.path.basename(image_path)}")
            
            try:
                if detect_single_image(model, image_path, output_dir, device, image_size):
                    success_count += 1
                    pbar.set_postfix({"Success": success_count})
                else:
                    pbar.set_postfix({"Failed": "Preprocessing error"})
            except Exception as e:
                print(f"❌ Error processing image {image_path}: {e}")
                pbar.set_postfix({"Failed": str(e)})
            
            pbar.update(1)
    
    print(f"\n✅ Folder detection completed! Successfully processed {success_count}/{len(image_files)} images")


def load_model(checkpoint_path: str, device):
    """Load SAM2-UNet model"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"Target device: {device}")
    
    model = SAM2UNet(device=device)
    
    try:
        # Method 1: Load directly to target device
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint, strict=True)
        print("✅ Checkpoint loaded successfully")
    except Exception as e1:
        print(f"Method 1 failed: {e1}")
        try:
            # Method 2: Load to CPU first, then move to target device
            print("Trying method 2: Load to CPU first...")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint, strict=True)
            print("✅ Checkpoint loaded successfully from CPU")
        except Exception as e2:
            print(f"Method 2 also failed: {e2}")
            print("Trying method 3: Ignore strict check...")
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
                if missing_keys:
                    print(f"⚠️ Missing keys: {len(missing_keys)}")
                if unexpected_keys:
                    print(f"⚠️ Unexpected keys: {len(unexpected_keys)}")
                print("✅ Checkpoint loaded successfully in non-strict mode")
            except Exception as e3:
                print(f"❌ All methods failed: {e3}")
                raise e3
    
    model.eval()
    model.to(device)
    print(f"✅ Model moved to device: {device}")
    return model


# Main program
def main():
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Check if input is file or folder
    if os.path.isfile(args.input):
        # Single file detection
        print(f"Starting single file detection: {args.input}")
        
        with tqdm(total=1, desc="Single file detection", unit="file") as pbar:
            pbar.set_description(f"Detecting file: {os.path.basename(args.input)}")
            
            success = detect_single_image(model, args.input, args.output, device, args.image_size)
            
            if success:
                pbar.set_postfix({"Status": "Success"})
            else:
                pbar.set_postfix({"Status": "Failed"})
            
            pbar.update(1)
            
    elif os.path.isdir(args.input):
        # Folder detection
        print(f"Starting folder detection: {args.input}")
        print(f"Using TestDataset: {args.use_dataset}")
        
        if args.use_dataset:
            # Use original TestDataset method
            detect_folder_with_dataset(model, args.input, args.output, device, args.image_size)
        else:
            # Use new folder detection method
            detect_folder_new(model, args.input, args.output, device, args.image_size)
        
    else:
        print(f"❌ Input path does not exist: {args.input}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())