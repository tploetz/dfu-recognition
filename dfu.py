
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DFU Detection Pipeline - Combining YOLO and SAM2-UNet for Diabetes Foot Ulcer Detection
1. Use YOLO for object detection
2. Use SAM2-UNet for precise segmentation
3. Combine results for final analysis
"""

import os
import sys
import argparse
import logging
import subprocess
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import json

# Import YOLO and SAM detection modules
try:
    from detect_yolo import YOLODetector
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: Cannot import YOLO detection module, will use command line mode")

try:
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    from PIL import Image
    import imageio
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("Warning: Cannot import SAM detection dependencies, will use command line mode")
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

# Configure standard output encoding
sys.stdout.reconfigure(encoding='utf-8')


class DFUDetector:
    """DFU Detector - Combining YOLO and SAM2-UNet"""
    
    def __init__(self, yolo_model_path: str, sam_model_path: str, device: str = "0"):
        """
        Initialize DFU detector
        
        Args:
            yolo_model_path: YOLO model path
            sam_model_path: SAM2-UNet model path
            device: Device ID
        """
        self.yolo_model_path = yolo_model_path
        self.sam_model_path = sam_model_path
        self.device = device
        
        # Check if model files exist
        if not os.path.exists(yolo_model_path):
            raise FileNotFoundError(f"YOLO model file not found: {yolo_model_path}")
        if not os.path.exists(sam_model_path):
            raise FileNotFoundError(f"SAM model file not found: {sam_model_path}")
        
        logger.info(f"DFU detector initialized")
        logger.info(f"YOLO model: {yolo_model_path}")
        logger.info(f"SAM model: {sam_model_path}")
        logger.info(f"Device: {device}")
    
    def run_yolo_detection(self, input_path: str, output_dir: str, 
                          conf_threshold: float = 0.25, save_yolo_txt: bool = True) -> Dict:
        """
        Run YOLO detection
        
        Args:
            input_path: Input path (file or folder)
            output_dir: Output directory
            conf_threshold: Confidence threshold
            save_yolo_txt: Whether to save YOLO format txt files
            
        Returns:
            Detection result dictionary
        """
        logger.info(f"Starting YOLO detection: {input_path}")
        
        # Prioritize direct function calls
        if YOLO_AVAILABLE:
            try:
                return self._run_yolo_direct(input_path, output_dir, conf_threshold, save_yolo_txt)
            except Exception as e:
                logger.warning(f"Direct YOLO call failed, falling back to command line mode: {e}")
        
        # Fallback to command line mode
        return self._run_yolo_subprocess(input_path, output_dir, conf_threshold, save_yolo_txt)
    
    def _run_yolo_direct(self, input_path: str, output_dir: str, 
                        conf_threshold: float = 0.25, save_yolo_txt: bool = True) -> Dict:
        """
        Direct YOLO detection function call
        """
        try:
            # Create YOLO detector
            detector = YOLODetector(self.yolo_model_path)
            
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            if os.path.isfile(input_path):
                # Single file detection
                result = detector.detect_single_image(
                    input_path, output_dir, conf_threshold, save_yolo_txt
                )
                logger.info("✅ YOLO single file detection completed")
                return {"success": True, "result": result}
            else:
                # Folder detection
                results = detector.detect_folder(
                    input_path, output_dir, conf_threshold, save_yolo_txt
                )
                logger.info(f"✅ YOLO folder detection completed, processed {len(results)} images")
                return {"success": True, "results": results}
                
        except Exception as e:
            logger.error(f"❌ YOLO direct call failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _run_yolo_subprocess(self, input_path: str, output_dir: str, 
                            conf_threshold: float = 0.25, save_yolo_txt: bool = True) -> Dict:
        """
        Use subprocess to call YOLO detection
        """
        # Build YOLO detection command
        cmd = [
            "python", "detect_yolo.py",
            "--model", self.yolo_model_path,
            "--input", input_path,
            "--output", output_dir,
            "--device", f"cuda:{self.device}" if self.device.isdigit() else "cpu",
            "--conf", str(conf_threshold)
        ]
        
        if save_yolo_txt:
            cmd.append("--save_yolo_txt")
        
        # Execute YOLO detection
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            if result.returncode == 0:
                logger.info("✅ YOLO detection completed")
                return {"success": True, "output_dir": output_dir}
            else:
                logger.error(f"❌ YOLO detection failed: {result.stderr}")
                return {"success": False, "error": result.stderr}
        except Exception as e:
            logger.error(f"❌ YOLO detection exception: {e}")
            return {"success": False, "error": str(e)}
    
    def run_sam_detection(self, input_path: str, output_dir: str, 
                         use_dataset: bool = True) -> Dict:
        """
        Run SAM2-UNet detection
        
        Args:
            input_path: Input path (file or folder)
            output_dir: Output directory
            use_dataset: Whether to use TestDataset approach
            
        Returns:
            Detection result dictionary
        """
        logger.info(f"Starting SAM2-UNet detection: {input_path}")
        
        # Prioritize direct function calls
        if SAM_AVAILABLE:
            try:
                return self._run_sam_direct(input_path, output_dir, use_dataset)
            except Exception as e:
                logger.warning(f"Direct SAM call failed, falling back to command line mode: {e}")
        
        # Fallback to command line mode
        return self._run_sam_subprocess(input_path, output_dir, use_dataset)
    
    def _run_sam_direct(self, input_path: str, output_dir: str, use_dataset: bool = True) -> Dict:
        """
        Direct SAM2-UNet detection function call
        """
        try:
            # Import SAM related modules - try different import paths
            try:
                from sam2_unet.SAM2UNet import SAM2UNet
            except ImportError:
                try:
                    from sam2_unet.model import SAM2UNet
                except ImportError:
                    try:
                        from SAM2UNet import SAM2UNet  # If directly in root directory
                    except ImportError:
                        # Try importing from current directory
                        import sys
                        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                        from sam2_unet.SAM2UNet import SAM2UNet
            
            # Set device
            device = torch.device(f"cuda:{self.device}" if self.device.isdigit() else "cpu")
            logger.info(f"Using device: {device}")
            
            # Load model
            model = SAM2UNet(device=device)
            
            # Try multiple loading methods (consistent with detect_sam.py)
            try:
                # Method 1: Load directly to target device
                checkpoint = torch.load(self.sam_model_path, map_location=device)
                model.load_state_dict(checkpoint, strict=True)
                logger.info("✅ Checkpoint loaded successfully")
            except Exception as e1:
                logger.warning(f"Method 1 failed: {e1}")
                try:
                    # Method 2: Load to CPU first, then move to target device
                    logger.info("Trying method 2: Load to CPU first...")
                    checkpoint = torch.load(self.sam_model_path, map_location='cpu')
                    model.load_state_dict(checkpoint, strict=True)
                    logger.info("✅ Checkpoint loaded successfully from CPU")
                except Exception as e2:
                    logger.warning(f"Method 2 also failed: {e2}")
                    logger.info("Trying method 3: Ignore strict check...")
                    try:
                        checkpoint = torch.load(self.sam_model_path, map_location='cpu')
                        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
                        if missing_keys:
                            logger.warning(f"⚠️ Missing keys: {len(missing_keys)}")
                        if unexpected_keys:
                            logger.warning(f"⚠️ Unexpected keys: {len(unexpected_keys)}")
                        logger.info("✅ Checkpoint loaded successfully in non-strict mode")
                    except Exception as e3:
                        raise Exception(f"All loading methods failed: {e3}")
            
            model = model.to(device)
            model.eval()
            
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            if os.path.isfile(input_path):
                # Single file detection
                result = self._predict_single_sam(model, device, input_path, output_dir)
                logger.info("✅ SAM single file detection completed")
                return {"success": True, "result": result}
            else:
                # Folder detection
                if use_dataset:
                    results = self._predict_folder_with_dataset(model, device, input_path, output_dir)
                else:
                    results = self._predict_folder_simple(model, device, input_path, output_dir)
                logger.info(f"✅ SAM folder detection completed, processed {len(results)} images")
                return {"success": True, "results": results}
                
        except Exception as e:
            logger.error(f"❌ SAM direct call failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _run_sam_subprocess(self, input_path: str, output_dir: str, use_dataset: bool = True) -> Dict:
        """
        Use subprocess to call SAM2-UNet detection
        """
        # Build SAM detection command
        cmd = [
            "python", "detect_sam.py",
            "--checkpoint", self.sam_model_path,
            "--input", input_path,
            "--output", output_dir,
            "--cuda_idx", self.device
        ]
        
        if use_dataset:
            cmd.append("--use_dataset")
        
        # Execute SAM detection
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            if result.returncode == 0:
                logger.info("✅ SAM2-UNet detection completed")
                return {"success": True, "output_dir": output_dir}
            else:
                logger.error(f"❌ SAM2-UNet detection failed: {result.stderr}")
                return {"success": False, "error": result.stderr}
        except Exception as e:
            logger.error(f"❌ SAM2-UNet detection exception: {e}")
            return {"success": False, "error": str(e)}
    
    def _predict_single_sam(self, model, device, image_path: str, output_dir: str) -> Dict:
        """
        Single file SAM prediction
        """
        try:
            # Preprocess image
            image = self._preprocess_image(image_path)
            original_shape = image.shape[:2]  # (height, width)
            
            # Convert to tensor
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
            image_tensor = image_tensor.to(device)
            
            # Prediction
            with torch.no_grad():
                res, _, _ = model(image_tensor)
                res = torch.sigmoid(res)
                
                # Resize to original dimensions
                res = F.upsample(res, size=original_shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu()
                res = res.numpy().squeeze()
                
                # Normalize to 0-255
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                res = (res * 255).astype(np.uint8)
            
            # Save result
            base_name = Path(image_path).stem
            output_path = os.path.join(output_dir, f"{base_name}_sam_mask.png")
            imageio.imsave(output_path, res)
            
            return {
                "image_path": image_path,
                "output_path": output_path,
                "original_shape": original_shape
            }
            
        except Exception as e:
            logger.error(f"Single file SAM prediction failed: {e}")
            return None
    
    def _predict_folder_with_dataset(self, model, device, folder_path: str, output_dir: str) -> List[Dict]:
        """
        Use dataset loader for folder prediction
        """
        try:
            # Create simple test dataset
            class SimpleTestDataset:
                def __init__(self, image_root, size):
                    self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    self.images = sorted(self.images)
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
                    original_shape = image.size[::-1]  # (height, width)
                    image = self.transform(image).unsqueeze(0)
                    name = os.path.basename(self.images[self.index])
                    self.index += 1
                    return image, original_shape, name
            
            test_loader = SimpleTestDataset(folder_path, 352)
            results = []
            
            with tqdm(total=test_loader.size, desc="SAM2-UNet detection progress", unit="images") as pbar:
                for i in range(test_loader.size):
                    image, gt_shape, name = test_loader.load_data()
                    image = image.to(device)
                    
                    with torch.no_grad():
                        res, _, _ = model(image)
                        res = torch.sigmoid(res)
                        res = F.upsample(res, size=gt_shape, mode='bilinear', align_corners=False)
                        res = res.sigmoid().data.cpu()
                        res = res.numpy().squeeze()
                        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                        res = (res * 255).astype(np.uint8)
                        
                        output_filename = name[:-4] + "_sam_mask.png"
                        output_path = os.path.join(output_dir, output_filename)
                        imageio.imsave(output_path, res)
                        
                        results.append({
                            "image_name": name,
                            "output_path": output_path,
                            "original_shape": gt_shape
                        })
                        
                        pbar.update(1)
            
            return results
            
        except Exception as e:
            logger.error(f"Dataset method SAM prediction failed: {e}")
            return []
    
    def _predict_folder_simple(self, model, device, folder_path: str, output_dir: str) -> List[Dict]:
        """
        Simple folder prediction method
        """
        try:
            image_files = [f for f in os.listdir(folder_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            results = []
            
            with tqdm(total=len(image_files), desc="SAM2-UNet detection progress", unit="images") as pbar:
                for image_file in image_files:
                    image_path = os.path.join(folder_path, image_file)
                    result = self._predict_single_sam(model, device, image_path, output_dir)
                    if result:
                        results.append(result)
                    pbar.update(1)
            
            return results
            
        except Exception as e:
            logger.error(f"Simple method SAM prediction failed: {e}")
            return []
    
    def _preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess image
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot read image: {image_path}")
            
            # Convert color space
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to 352x352
            image = cv2.resize(image, (352, 352))
            
            return image
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return None
    
    def draw_bbox_from_yolo(self, image_path: str, yolo_txt_path: str, save_path: str):
        """
        Draw bounding box mask based on YOLO results
        
        Args:
            image_path: Original image path
            yolo_txt_path: YOLO label file path
            save_path: Save path
        """
        # Read original image to get dimensions
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"Cannot read image: {image_path}")
            return False
            
        height, width = image.shape[:2]
        bbox_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Check if YOLO label file exists
        if os.path.exists(yolo_txt_path):
            with open(yolo_txt_path, 'r') as file:
                lines = file.readlines()
            
            # Process each bounding box
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    x_center, y_center, bbox_width, bbox_height = map(float, parts[1:5])
                    
                    # Convert from YOLO format to pixel coordinates
                    x_min = (x_center - bbox_width / 2) * width
                    y_min = (y_center - bbox_height / 2) * height
                    x_max = (x_center + bbox_width / 2) * width
                    y_max = (y_center + bbox_height / 2) * height
                    
                    # Draw bounding box mask
                    cv2.rectangle(bbox_mask, (int(x_min), int(y_min)), (int(x_max), int(y_max)), 255, -1)
        
        # Save bounding box mask
        cv2.imwrite(save_path, bbox_mask)
        return True
    
    def overlap_crop_gt(self, sam_mask: np.ndarray, yolo_mask: np.ndarray) -> np.ndarray:
        """
        Calculate intersection of SAM segmentation mask and YOLO bounding box mask
        
        Args:
            sam_mask: SAM segmentation mask
            yolo_mask: YOLO bounding box mask
            
        Returns:
            Intersection mask
        """
        # Binary processing
        _, sam_mask_binary = cv2.threshold(sam_mask, 128, 1, cv2.THRESH_BINARY)
        sam_mask_binary = sam_mask_binary.astype(np.uint8)
        
        # Find contours of SAM mask
        contours, _ = cv2.findContours(sam_mask_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        output = np.zeros_like(yolo_mask)
        
        if len(contours) == 0:
            return output
        
        # Calculate intersection with YOLO mask for each contour
        for i in range(len(contours)):
            # Create mask for single contour
            contour_mask = np.zeros(sam_mask.shape, dtype=np.uint8)
            cv2.drawContours(contour_mask, contours, i, 255, -1)
            
            # Calculate intersection with YOLO mask
            overlap = cv2.bitwise_and(contour_mask, yolo_mask)
            
            if np.sum(overlap) > 0:
                output += contour_mask
        
        return output
    
    def generate_final_results(self, input_path: str, yolo_output_dir: str, 
                              sam_output_dir: str, final_output_dir: str) -> Dict:
        """
        Generate final recognition results
        
        Args:
            input_path: Original input path
            yolo_output_dir: YOLO output directory
            sam_output_dir: SAM output directory
            final_output_dir: Final output directory
            
        Returns:
            Processing result dictionary
        """
        logger.info("Starting to generate final recognition results")
        
        # Create output directory structure
        os.makedirs(final_output_dir, exist_ok=True)
        visualization_dir = os.path.join(os.path.dirname(final_output_dir), "visualization")
        yolo_bbox_dir = os.path.join(os.path.dirname(final_output_dir), "yolo_bbox")
        os.makedirs(visualization_dir, exist_ok=True)
        os.makedirs(yolo_bbox_dir, exist_ok=True)
        
        # Get all original image files
        image_files = []
        if os.path.isfile(input_path):
            image_files = [os.path.basename(input_path)]
            input_dir = os.path.dirname(input_path)
        else:
            input_dir = input_path
            for root, dirs, files in os.walk(input_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_files.append(file)
        
        results = {
            "total_images": len(image_files),
            "processed_images": 0,
            "results": []
        }
        
        with tqdm(total=len(image_files), desc="Generating final results", unit="images") as pbar:
            for image_file in image_files:
                try:
                    base_name = Path(image_file).stem
                    
                    # File paths
                    original_image_path = os.path.join(input_dir, image_file)
                    yolo_txt_path = os.path.join(yolo_output_dir, f"{base_name}.txt")
                    sam_mask_path = os.path.join(sam_output_dir, f"{base_name}_sam_mask.png")
                    
                    result = {
                        "image_name": image_file,
                        "yolo_detections": 0,
                        "sam_mask_exists": False,
                        "overlap_exists": False
                    }
                    
                    # Check if files exist
                    if not os.path.exists(original_image_path):
                        logger.warning(f"Original image does not exist: {original_image_path}")
                        continue
                    
                    # 1. Draw YOLO bounding box mask (save to yolo_bbox directory)
                    yolo_mask_path = os.path.join(yolo_bbox_dir, f"{base_name}_yolo_bbox.png")
                    if self.draw_bbox_from_yolo(original_image_path, yolo_txt_path, yolo_mask_path):
                        result["yolo_detections"] = 1  # Simplified count
                    
                    # 2. Read SAM segmentation mask
                    sam_mask = None
                    if os.path.exists(sam_mask_path):
                        sam_mask = cv2.imread(sam_mask_path, cv2.IMREAD_GRAYSCALE)
                        result["sam_mask_exists"] = True
                    
                    # 3. Calculate intersection
                    if sam_mask is not None and os.path.exists(yolo_mask_path):
                        yolo_mask = cv2.imread(yolo_mask_path, cv2.IMREAD_GRAYSCALE)
                        
                        # Calculate intersection
                        overlap_mask = self.overlap_crop_gt(sam_mask, yolo_mask)
                        
                        # Save intersection mask (final result, save to combined_results directory)
                        overlap_path = os.path.join(final_output_dir, f"{base_name}_final_mask.png")
                        cv2.imwrite(overlap_path, overlap_mask)
                        
                        result["overlap_exists"] = True
                        
                        # 4. Generate visualization results (save to visualization directory)
                        self.create_visualization(original_image_path, yolo_mask, sam_mask, 
                                                overlap_mask, visualization_dir, base_name)
                    
                    results["results"].append(result)
                    results["processed_images"] += 1
                    
                    pbar.set_postfix({
                        "YOLO": result["yolo_detections"],
                        "SAM": "Yes" if result["sam_mask_exists"] else "No",
                        "Overlap": "Yes" if result["overlap_exists"] else "No"
                    })
                    pbar.update(1)
                    
                except Exception as e:
                    logger.error(f"Error processing image {image_file}: {e}")
                    continue
        
        logger.info(f"✅ Final results generation completed, processed {results['processed_images']} images")
        return results
    
    def create_visualization(self, original_image_path: str, yolo_mask: np.ndarray, 
                           sam_mask: np.ndarray, overlap_mask: np.ndarray,
                           output_dir: str, base_name: str):
        """
        Create visualization results
        
        Args:
            original_image_path: Original image path
            yolo_mask: YOLO bounding box mask
            sam_mask: SAM segmentation mask
            overlap_mask: Intersection mask
            output_dir: Output directory
            base_name: Base filename
        """
        try:
            # Read original image
            original = cv2.imread(original_image_path)
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            
            # Create colored masks
            yolo_colored = cv2.applyColorMap(yolo_mask, cv2.COLORMAP_JET)
            sam_colored = cv2.applyColorMap(sam_mask, cv2.COLORMAP_HOT)
            overlap_colored = cv2.applyColorMap(overlap_mask, cv2.COLORMAP_VIRIDIS)
            
            # Overlay on original image
            yolo_overlay = cv2.addWeighted(original, 0.7, yolo_colored, 0.3, 0)
            sam_overlay = cv2.addWeighted(original, 0.7, sam_colored, 0.3, 0)
            overlap_overlay = cv2.addWeighted(original, 0.7, overlap_colored, 0.3, 0)
            
            # Save visualization results
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_yolo_visual.png"), 
                       cv2.cvtColor(yolo_overlay, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_sam_visual.png"), 
                       cv2.cvtColor(sam_overlay, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_overlap_visual.png"), 
                       cv2.cvtColor(overlap_overlay, cv2.COLOR_RGB2BGR))
            
        except Exception as e:
            logger.warning(f"Failed to create visualization results: {e}")
    
    def run_full_detection(self, input_path: str, output_base_dir: str,
                          yolo_conf: float = 0.25, save_yolo_txt: bool = True,
                          use_sam_dataset: bool = True) -> Dict:
        """
        Run complete DFU detection pipeline
        
        Args:
            input_path: Input path (file or folder)
            output_base_dir: Output base directory
            yolo_conf: YOLO confidence threshold
            save_yolo_txt: Whether to save YOLO format txt files
            use_sam_dataset: Whether to use SAM's TestDataset method
            
        Returns:
            Complete detection results
        """
        logger.info("=" * 60)
        logger.info("Starting complete DFU detection pipeline")
        logger.info("=" * 60)
        
        # Create output directories
        yolo_output_dir = os.path.join(output_base_dir, "yolo_results")
        sam_output_dir = os.path.join(output_base_dir, "sam_results")
        final_output_dir = os.path.join(output_base_dir, "combined_results")
        
        os.makedirs(yolo_output_dir, exist_ok=True)
        os.makedirs(sam_output_dir, exist_ok=True)
        os.makedirs(final_output_dir, exist_ok=True)
        
        full_results = {
            "input_path": input_path,
            "output_directories": {
                "yolo": yolo_output_dir,
                "sam": sam_output_dir,
                "combined": final_output_dir
            },
            "steps": {}
        }
        
        # Step 1: YOLO detection
        logger.info("Step 1: YOLO object detection")
        yolo_result = self.run_yolo_detection(
            input_path, yolo_output_dir, 
            conf_threshold=yolo_conf, save_yolo_txt=save_yolo_txt
        )
        full_results["steps"]["yolo"] = yolo_result
        
        if not yolo_result["success"]:
            logger.error("YOLO detection failed, terminating pipeline")
            return full_results
        
        # Step 2: SAM2-UNet segmentation
        logger.info("Step 2: SAM2-UNet precise segmentation")
        sam_result = self.run_sam_detection(
            input_path, sam_output_dir,
            use_dataset=use_sam_dataset
        )
        full_results["steps"]["sam"] = sam_result
        
        if not sam_result["success"]:
            logger.error("SAM detection failed, terminating pipeline")
            return full_results
        
        # Step 3: Generate final recognition results
        logger.info("Step 3: Generate final recognition results (YOLO bounding box + SAM segmentation intersection)")
        final_result = self.generate_final_results(
            input_path, yolo_output_dir, sam_output_dir, final_output_dir
        )
        full_results["steps"]["final"] = final_result
        
        logger.info("=" * 60)
        logger.info("✅ Complete DFU detection pipeline completed")
        logger.info(f"Results saved in: {output_base_dir}")
        logger.info("=" * 60)
        
        return full_results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='DFU Detection Pipeline - Combining YOLO and SAM2-UNet')
    
    # Required parameters
    parser.add_argument('--yolo_model', type=str, default='models/runs/train/exp2/weights/best.pt',
                       help='YOLO model path')
    parser.add_argument('--sam_model', type=str, default='models/1-SAM2-UNet-20.pth',
                       help='SAM2-UNet model path')
    parser.add_argument('--input', '-i', type=str, required=False, default='data/DFU_split/ulcer/test/images',
                       help='Input path (file or folder)')
    parser.add_argument('--output', '-o', type=str, required=False, default='prediction/dfu',
                       help='Output directory path')
    
    # Optional parameters
    parser.add_argument('--device', type=str, default='0',
                       help='Device ID (default: 0)')
    parser.add_argument('--yolo_conf', type=float, default=0.25,
                       help='YOLO confidence threshold (default: 0.25)')
    parser.add_argument('--save_yolo_txt', action='store_true', default=True,
                       help='Save YOLO format txt files')
    parser.add_argument('--use_sam_dataset', action='store_true', default=True,
                       help='Use SAM TestDataset method')
    
    args = parser.parse_args()
    
    try:
        # Create DFU detector
        detector = DFUDetector(
            yolo_model_path=args.yolo_model,
            sam_model_path=args.sam_model,
            device=args.device
        )
        
        # Run complete detection pipeline
        results = detector.run_full_detection(
            input_path=args.input,
            output_base_dir=args.output,
            yolo_conf=args.yolo_conf,
            save_yolo_txt=args.save_yolo_txt,
            use_sam_dataset=args.use_sam_dataset
        )
        
        # Print result summary
        if results["steps"]["final"]["processed_images"] > 0:
            logger.info(f"Detection completed! Processed {results['steps']['final']['processed_images']} images")
            logger.info(f"Results saved in: {args.output}")
            logger.info("Directory structure:")
            logger.info(f"  - {args.output}/combined_results/     # Final recognition result masks")
            logger.info(f"  - {args.output}/visualization/        # Visualization results")
            logger.info(f"  - {args.output}/yolo_bbox/           # YOLO bounding box masks")
            logger.info("File types:")
            logger.info("  - *_final_mask.png     # Final recognition results")
            logger.info("  - *_yolo_visual.png    # YOLO visualization")
            logger.info("  - *_sam_visual.png     # SAM visualization")
            logger.info("  - *_overlap_visual.png # Intersection visualization")
        else:
            logger.warning("No images were processed")
            
    except Exception as e:
        logger.error(f"DFU detection pipeline error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
