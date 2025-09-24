#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO Detection Tool - Supports both folder and single file calling methods
Optimized for DUCSS-DFU-Recognition project
Supports multiple YOLO models: YOLOv8, YOLOv9, YOLOv10, YOLOv11, YOLOWorld, YOLOE
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple
import json
import cv2
import numpy as np
from datetime import datetime
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


class YOLODetector:
    """YOLO detector class, supports multiple YOLO models and calling methods"""
    
    def __init__(self, model_path: str, model_type: str = "auto", device: str = ""):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Model file path
            model_type: Model type ("yolo", "yoloworld", "yoloe", "auto")
            device: Device type ("", "cpu", "cuda", "cuda:0", etc.)
        """
        self.model_path = model_path
        self.model_type = model_type
        self.device = device
        self.model = None
        self.class_names = None
        
        # Supported image formats
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model"""
        try:
            # Handle relative paths based on project root directory
            if not os.path.isabs(self.model_path):
                # If it's a relative path, start from project root directory
                project_root = os.path.dirname(os.path.abspath(__file__))
                self.model_path = os.path.join(project_root, self.model_path)
            
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file does not exist: {self.model_path}")
            
            # Auto-detect model type
            if self.model_type == "auto":
                self.model_type = self._detect_model_type()
            
            logger.info(f"Loading {self.model_type} model: {self.model_path}")
            
            # Set device
            if self.device:
                if self.device.startswith('cuda') and not self._check_cuda():
                    logger.warning("CUDA not available, using CPU")
                    self.device = 'cpu'
            
            # Try multiple ways to load model
            try:
                if self.model_type == "yoloworld":
                    from ultralytics import YOLOWorld
                    self.model = YOLOWorld(self.model_path)
                elif self.model_type == "yoloe":
                    from ultralytics import YOLOE
                    self.model = YOLOE(self.model_path)
                else:  # Default to standard YOLO
                    from ultralytics import YOLO
                    self.model = YOLO(self.model_path)
            except Exception as model_error:
                logger.warning(f"Standard loading method failed: {model_error}")
                logger.info("Trying to load model using compatibility mode...")
                
                # Compatibility mode: Try loading pre-trained model first, then load weights
                try:
                    from ultralytics import YOLO
                    # Use a general pre-trained model as base
                    base_model = "yolo11n.pt"  # or "yolov8n.pt"
                    logger.info(f"Using base model: {base_model}")
                    self.model = YOLO(base_model)
                    
                    # Try loading custom weights
                    import torch
                    checkpoint = torch.load(self.model_path, map_location='cpu')
                    self.model.model.load_state_dict(checkpoint['model'].state_dict(), strict=False)
                    logger.info("Successfully loaded custom weights")
                    
                except Exception as compat_error:
                    logger.error(f"Compatibility mode also failed: {compat_error}")
                    # Last attempt: Use pre-trained model directly
                    logger.info("Using pre-trained model for detection...")
                    self.model = YOLO("yolo11n.pt")
                    logger.warning("Note: Using pre-trained model, not custom trained DFU model")
            
            logger.info(f"Model loaded successfully, using device: {self.device if self.device else 'auto'}")
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise
    
    def _detect_model_type(self) -> str:
        """Auto-detect model type"""
        model_name = os.path.basename(self.model_path).lower()
        
        if 'world' in model_name or 'yoloworld' in model_name:
            return "yoloworld"
        elif 'yoloe' in model_name or 'e' in model_name:
            return "yoloe"
        else:
            return "yolo"
    
    def _check_cuda(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def set_classes(self, class_names: List[str]):
        """Set class names (mainly for YOLOWorld)"""
        if self.model_type == "yoloworld":
            self.model.set_classes(class_names)
            self.class_names = class_names
            logger.info(f"Set classes: {class_names}")
        else:
            logger.warning("Only YOLOWorld model supports dynamic class setting")
    
    def detect_folder(self, folder_path: str, output_dir: str = None, 
                     save_results: bool = True, save_images: bool = True,
                     save_yolo_txt: bool = True, save_json: bool = False,
                     conf_threshold: float = 0.25, iou_threshold: float = 0.7,
                     max_det: int = 1000) -> Dict:
        """
        Detect all images in a folder
        
        Args:
            folder_path: Input folder path
            output_dir: Output directory path
            save_results: Whether to save detection results
            save_images: Whether to save annotated images
            save_yolo_txt: Whether to save YOLO format txt label files
            save_json: Whether to save JSON format detailed results
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold
            max_det: Maximum detection count
            
        Returns:
            Detection results dictionary
        """
        # Handle relative paths
        if not os.path.isabs(folder_path):
            project_root = os.path.dirname(os.path.abspath(__file__))
            folder_path = os.path.join(project_root, folder_path)
        
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder does not exist: {folder_path}")
        
        # Get all image files
        image_files = self._get_image_files(folder_path)
        if not image_files:
            logger.warning(f"No supported image files found in folder {folder_path}")
            return {}
        
        logger.info(f"Found {len(image_files)} image files")
        
        # Set output directory
        if output_dir is None:
            output_dir = os.path.join(folder_path, "yolo_detection_results")
        
        # Handle relative paths
        if not os.path.isabs(output_dir):
            project_root = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(project_root, output_dir)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Detection results
        all_results = {
            "detection_info": {
                "model_path": self.model_path,
                "model_type": self.model_type,
                "input_folder": folder_path,
                "output_folder": output_dir,
                "total_images": len(image_files),
                "processed_images": 0,
                "detection_time": None
            },
            "results": []
        }
        
        start_time = datetime.now()
        
        # Batch detection - using progress bar
        with tqdm(total=len(image_files), desc="Detection progress", unit="images", 
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            for i, image_path in enumerate(image_files):
                try:
                    # Update progress bar description
                    pbar.set_description(f"Detection progress: {os.path.basename(image_path)}")
                    
                    # Detect single image
                    result = self.detect_single_image(
                        image_path, 
                        conf_threshold=conf_threshold,
                        iou_threshold=iou_threshold,
                        max_det=max_det
                    )
                    
                    if result:
                        # Save results
                        if save_results:
                            if save_yolo_txt:
                                self._save_yolo_txt(result, output_dir)
                            if save_json:
                                self._save_json_result(result, output_dir)
                        
                        # Save annotated images
                        if save_images:
                            self._save_annotated_image(result, output_dir)
                        
                        all_results["results"].append(result)
                        all_results["detection_info"]["processed_images"] += 1
                        
                        # Update progress bar post-processing information
                        pbar.set_postfix({
                            "Detected": result['total_detections'],
                            "Success": all_results["detection_info"]["processed_images"]
                        })
                    
                    # Update progress bar
                    pbar.update(1)
                
                except Exception as e:
                    logger.error(f"Error processing image {image_path}: {e}")
                    pbar.update(1)
                    continue
        
        # Calculate total time
        end_time = datetime.now()
        all_results["detection_info"]["detection_time"] = str(end_time - start_time)
        
        # Save summary results
        if save_results:
            summary_file = os.path.join(output_dir, "detection_summary.json")
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            logger.info(f"Detection summary results saved: {summary_file}")
        
        logger.info(f"Folder detection completed! Processed {all_results['detection_info']['processed_images']} images")
        return all_results
    
    def detect_single_image(self, image_path: str, conf_threshold: float = 0.25,
                           iou_threshold: float = 0.7, max_det: int = 1000) -> Dict:
        """
        Detect single image
        
        Args:
            image_path: Image file path
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold
            max_det: Maximum detection count
            
        Returns:
            Detection results dictionary
        """
        # Handle relative paths
        if not os.path.isabs(image_path):
            project_root = os.path.dirname(os.path.abspath(__file__))
            image_path = os.path.join(project_root, image_path)
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file does not exist: {image_path}")
        
        try:
            # Execute detection
            results = self.model.predict(
                image_path,
                conf=conf_threshold,
                iou=iou_threshold,
                max_det=max_det,
                device=self.device,
                verbose=False
            )
            
            if not results:
                logger.warning(f"No objects detected: {image_path}")
                return None
            
            result = results[0]
            
            # Extract detection information
            detection_result = {
                "image_path": image_path,
                "image_name": os.path.basename(image_path),
                "image_size": result.orig_shape,
                "detections": []
            }
            
            # Extract bounding boxes and class information
            if result.boxes is not None:
                boxes = result.boxes.data.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2, conf, cls_id = box[:6]
                    class_name = result.names[int(cls_id)] if hasattr(result, 'names') else f"class_{int(cls_id)}"
                    
                    detection_result["detections"].append({
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": float(conf),
                        "class_id": int(cls_id),
                        "class_name": class_name
                    })
            
            detection_result["total_detections"] = len(detection_result["detections"])
            
            return detection_result
            
        except Exception as e:
            logger.error(f"Error detecting image {image_path}: {e}")
            return None
    
    def _get_image_files(self, folder_path: str) -> List[str]:
        """Get all image files in the folder"""
        image_files = []
        
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in self.supported_formats):
                    image_files.append(os.path.join(root, file))
        
        return sorted(image_files)
    
    def _save_yolo_txt(self, result: Dict, output_dir: str):
        """Save YOLO format txt label file"""
        txt_file = os.path.join(
            output_dir, 
            f"{Path(result['image_name']).stem}.txt"
        )
        
        # Get image dimensions
        img_width, img_height = result['image_size'][:2]
        
        with open(txt_file, 'w', encoding='utf-8') as f:
            for detection in result['detections']:
                # YOLO format: class_id center_x center_y width height
                x1, y1, x2, y2 = detection['bbox']
                
                # Calculate center point and width/height (normalized coordinates)
                center_x = (x1 + x2) / 2.0 / img_width
                center_y = (y1 + y2) / 2.0 / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                
                # Write YOLO format label
                f.write(f"{detection['class_id']} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
    
    def _save_json_result(self, result: Dict, output_dir: str):
        """Save JSON format detailed results"""
        json_file = os.path.join(
            output_dir, 
            f"{Path(result['image_name']).stem}_detection.json"
        )
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    
    def _save_annotated_image(self, result: Dict, output_dir: str):
        """Save annotated image"""
        try:
            # Read original image
            image = cv2.imread(result['image_path'])
            if image is None:
                logger.warning(f"Cannot read image: {result['image_path']}")
                return
            
            # Draw detection boxes
            for detection in result['detections']:
                x1, y1, x2, y2 = map(int, detection['bbox'])
                conf = detection['confidence']
                class_name = detection['class_name']
                
                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"{class_name}: {conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(image, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Save annotated image
            output_file = os.path.join(
                output_dir, 
                f"{Path(result['image_name']).stem}_annotated.jpg"
            )
            cv2.imwrite(output_file, image)
            
        except Exception as e:
            logger.error(f"Error saving annotated image: {e}")
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            "model_path": self.model_path,
            "model_type": self.model_type,
            "device": self.device,
            "class_names": self.class_names
        }


def main():
    """Main function - command line interface"""
    parser = argparse.ArgumentParser(description='YOLO Detection Tool - DUCSS-DFU-Recognition')
    
    # Required parameters
    parser.add_argument('--model', '-m', type=str, required=False, default="models/runs/train/exp2/weights/best.pt",
                       help='Model file path (supports relative paths, e.g.: models/dfu_yolo.pt)')
    parser.add_argument('--input', '-i', type=str, required=False, default="data/DFU_split/ulcer/test/images",
                       help='Input path (folder or image file, supports relative paths)')
    
    # Optional parameters
    parser.add_argument('--output', '-o', type=str, default="prediction/yolo/dfu_detection",
                       help='Output directory path (supports relative paths)')
    parser.add_argument('--model_type', type=str, default='auto',
                       choices=['auto', 'yolo', 'yoloworld', 'yoloe'],
                       help='Model type')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device type (cpu, cuda, cuda:0, etc.)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7,
                       help='IoU threshold')
    parser.add_argument('--max_det', type=int, default=1000,
                       help='Maximum detection count')
    parser.add_argument('--classes', type=str, nargs='*', default=None,
                       help='Class name list (for YOLOWorld)')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save results')
    parser.add_argument('--no_images', action='store_true',
                       help='Do not save annotated images')
    parser.add_argument('--save_yolo_txt', action='store_true', default=True,
                       help='Save YOLO format txt label files (enabled by default)')
    parser.add_argument('--save_json', action='store_true',
                       help='Save JSON format detailed results')
    
    args = parser.parse_args()
    
    try:
        # Create detector
        detector = YOLODetector(
            model_path=args.model,
            model_type=args.model_type,
            device=args.device
        )
        
        # Set classes (if provided)
        if args.classes:
            detector.set_classes(args.classes)
        
        # Check if input is file or folder
        if os.path.isfile(args.input):
            # Single file detection
            logger.info(f"Starting single file detection: {args.input}")
            
            # Single file detection progress display
            with tqdm(total=1, desc="Single file detection", unit="file") as pbar:
                pbar.set_description(f"Detecting file: {os.path.basename(args.input)}")
                
                result = detector.detect_single_image(
                    args.input,
                    conf_threshold=args.conf,
                    iou_threshold=args.iou,
                    max_det=args.max_det
                )
                
                if result:
                    pbar.set_postfix({"Detected objects": result['total_detections']})
                
                pbar.update(1)
            
            if result:
                logger.info(f"Detection completed, found {result['total_detections']} objects")
                if not args.no_save and args.output:
                    os.makedirs(args.output, exist_ok=True)
                    if args.save_yolo_txt:
                        detector._save_yolo_txt(result, args.output)
                    if args.save_json:
                        detector._save_json_result(result, args.output)
                    if not args.no_images:
                        detector._save_annotated_image(result, args.output)
            else:
                logger.warning("No objects detected")
                
        elif os.path.isdir(args.input):
            # Folder detection
            logger.info(f"Starting folder detection: {args.input}")
            results = detector.detect_folder(
                args.input,
                output_dir=args.output,
                save_results=not args.no_save,
                save_images=not args.no_images,
                save_yolo_txt=args.save_yolo_txt,
                save_json=args.save_json,
                conf_threshold=args.conf,
                iou_threshold=args.iou,
                max_det=args.max_det
            )
            
            if results and results['results']:
                total_detections = sum(r['total_detections'] for r in results['results'])
                logger.info(f"Folder detection completed, total detected {total_detections} objects")
            else:
                logger.warning("No objects detected in folder")
        else:
            logger.error(f"Input path does not exist: {args.input}")
            return 1
            
    except Exception as e:
        logger.error(f"Error during detection process: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
