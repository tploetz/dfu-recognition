"""
YOLO Training Utilities
Contains model analysis, result visualization, dataset statistics and other functions
"""

import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ultralytics import YOLO
import cv2
from collections import Counter
import json


class YOLOTrainingUtils:
    """YOLO Training Utilities Class"""
    
    def __init__(self, model_path=None, data_yaml=None):
        self.model_path = model_path
        self.data_yaml = data_yaml
        self.model = None
        self.data_config = None
        
        if model_path:
            self.load_model(model_path)
        if data_yaml:
            self.load_data_config(data_yaml)
    
    def load_model(self, model_path):
        """Load YOLO model"""
        try:
            self.model = YOLO(model_path)
            self.model_path = model_path
            print(f"✓ Model loaded successfully: {model_path}")
        except Exception as e:
            print(f"✗ Model loading failed: {e}")
            raise
    
    def load_data_config(self, data_yaml):
        """Load dataset configuration"""
        try:
            with open(data_yaml, 'r', encoding='utf-8') as f:
                self.data_config = yaml.safe_load(f)
            self.data_yaml = data_yaml
            print(f"✓ Dataset configuration loaded successfully: {data_yaml}")
        except Exception as e:
            print(f"✗ Dataset configuration loading failed: {e}")
            raise
    
    def analyze_dataset(self, save_plots=True, output_dir='dataset_analysis'):
        """Analyze dataset statistics"""
        if not self.data_config:
            raise ValueError("Please load dataset configuration first")
        
        print("Starting dataset analysis...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get dataset paths
        base_path = self.data_config['path']
        train_path = os.path.join(base_path, self.data_config['train'])
        val_path = os.path.join(base_path, self.data_config['val'])
        
        # Analyze training set
        train_stats = self._analyze_split(train_path, 'train')
        val_stats = self._analyze_split(val_path, 'val')
        
        # Generate statistical report
        self._generate_dataset_report(train_stats, val_stats, output_dir)
        
        # Plot statistical charts
        if save_plots:
            self._plot_dataset_stats(train_stats, val_stats, output_dir)
        
        return train_stats, val_stats
    
    def _analyze_split(self, split_path, split_name):
        """Analyze single dataset split"""
        images_path = os.path.join(split_path, 'images')
        labels_path = os.path.join(split_path, 'labels')
        
        if not os.path.exists(images_path) or not os.path.exists(labels_path):
            print(f"Warning: {split_name} dataset path does not exist")
            return None
        
        # Get all image and label files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']:
            image_files.extend(Path(images_path).glob(ext))
            image_files.extend(Path(images_path).glob(ext.upper()))
        
        label_files = list(Path(labels_path).glob('*.txt'))
        
        # Statistical information
        stats = {
            'split_name': split_name,
            'num_images': len(image_files),
            'num_labels': len(label_files),
            'image_sizes': [],
            'object_counts': [],
            'class_distribution': Counter(),
            'bbox_sizes': [],
            'bbox_aspect_ratios': []
        }
        
        # Analyze each image
        for img_file in image_files:
            # Get image size
            img = cv2.imread(str(img_file))
            if img is not None:
                h, w = img.shape[:2]
                stats['image_sizes'].append((w, h))
            
            # Get corresponding label file
            label_file = Path(labels_path) / (img_file.stem + '.txt')
            if label_file.exists():
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                object_count = len(lines)
                stats['object_counts'].append(object_count)
                
                # Analyze each annotation
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:5])
                        
                        stats['class_distribution'][class_id] += 1
                        
                        # Calculate bounding box size and aspect ratio
                        bbox_w = width * w if 'w' in locals() else width
                        bbox_h = height * h if 'h' in locals() else height
                        stats['bbox_sizes'].append((bbox_w, bbox_h))
                        stats['bbox_aspect_ratios'].append(bbox_w / bbox_h if bbox_h > 0 else 0)
        
        return stats
    
    def _generate_dataset_report(self, train_stats, val_stats, output_dir):
        """Generate dataset statistical report"""
        report_path = os.path.join(output_dir, 'dataset_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("YOLO Dataset Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Dataset basic information
            f.write("Dataset Basic Information:\n")
            f.write(f"Number of classes: {self.data_config.get('nc', 0)}\n")
            f.write(f"Class names: {self.data_config.get('names', [])}\n\n")
            
            # Training set statistics
            if train_stats:
                f.write("Training Set Statistics:\n")
                f.write(f"  Number of images: {train_stats['num_images']}\n")
                f.write(f"  Number of labels: {train_stats['num_labels']}\n")
                f.write(f"  Average objects per image: {np.mean(train_stats['object_counts']):.2f}\n")
                f.write(f"  Maximum objects: {max(train_stats['object_counts']) if train_stats['object_counts'] else 0}\n")
                f.write(f"  Minimum objects: {min(train_stats['object_counts']) if train_stats['object_counts'] else 0}\n")
                
                if train_stats['image_sizes']:
                    widths, heights = zip(*train_stats['image_sizes'])
                    f.write(f"  Average image size: {np.mean(widths):.0f}x{np.mean(heights):.0f}\n")
                    f.write(f"  Maximum image size: {max(widths)}x{max(heights)}\n")
                    f.write(f"  Minimum image size: {min(widths)}x{min(heights)}\n")
                
                f.write(f"  Class distribution: {dict(train_stats['class_distribution'])}\n\n")
            
            # Validation set statistics
            if val_stats:
                f.write("Validation Set Statistics:\n")
                f.write(f"  Number of images: {val_stats['num_images']}\n")
                f.write(f"  Number of labels: {val_stats['num_labels']}\n")
                f.write(f"  Average objects per image: {np.mean(val_stats['object_counts']):.2f}\n")
                f.write(f"  Maximum objects: {max(val_stats['object_counts']) if val_stats['object_counts'] else 0}\n")
                f.write(f"  Minimum objects: {min(val_stats['object_counts']) if val_stats['object_counts'] else 0}\n")
                
                if val_stats['image_sizes']:
                    widths, heights = zip(*val_stats['image_sizes'])
                    f.write(f"  Average image size: {np.mean(widths):.0f}x{np.mean(heights):.0f}\n")
                    f.write(f"  Maximum image size: {max(widths)}x{max(heights)}\n")
                    f.write(f"  Minimum image size: {min(widths)}x{min(heights)}\n")
                
                f.write(f"  Class distribution: {dict(val_stats['class_distribution'])}\n\n")
        
        print(f"Dataset report saved to: {report_path}")
    
    def _plot_dataset_stats(self, train_stats, val_stats, output_dir):
        """Plot dataset statistical charts"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Dataset Statistical Analysis', fontsize=16)
        
        # Image size distribution
        if train_stats and train_stats['image_sizes']:
            widths, heights = zip(*train_stats['image_sizes'])
            axes[0, 0].scatter(widths, heights, alpha=0.6, s=20)
            axes[0, 0].set_xlabel('Image Width')
            axes[0, 0].set_ylabel('Image Height')
            axes[0, 0].set_title('Training Set Image Size Distribution')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Object count distribution
        if train_stats and train_stats['object_counts']:
            axes[0, 1].hist(train_stats['object_counts'], bins=20, alpha=0.7, edgecolor='black')
            axes[0, 1].set_xlabel('Number of Objects per Image')
            axes[0, 1].set_ylabel('Number of Images')
            axes[0, 1].set_title('Training Set Object Count Distribution')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Class distribution
        if train_stats and train_stats['class_distribution']:
            classes = list(train_stats['class_distribution'].keys())
            counts = list(train_stats['class_distribution'].values())
            class_names = [self.data_config.get('names', [])[i] if i < len(self.data_config.get('names', [])) else f'Class_{i}' for i in classes]
            
            axes[0, 2].bar(class_names, counts)
            axes[0, 2].set_xlabel('Class')
            axes[0, 2].set_ylabel('Number of Objects')
            axes[0, 2].set_title('Training Set Class Distribution')
            axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Bounding box size distribution
        if train_stats and train_stats['bbox_sizes']:
            bbox_areas = [w * h for w, h in train_stats['bbox_sizes']]
            axes[1, 0].hist(bbox_areas, bins=50, alpha=0.7, edgecolor='black')
            axes[1, 0].set_xlabel('Bounding Box Area')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].set_title('Training Set Bounding Box Area Distribution')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Bounding box aspect ratio distribution
        if train_stats and train_stats['bbox_aspect_ratios']:
            axes[1, 1].hist(train_stats['bbox_aspect_ratios'], bins=50, alpha=0.7, edgecolor='black')
            axes[1, 1].set_xlabel('Aspect Ratio')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title('Training Set Bounding Box Aspect Ratio Distribution')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Training set vs validation set comparison
        if train_stats and val_stats:
            categories = ['Number of Images', 'Average Objects', 'Max Objects']
            train_values = [
                train_stats['num_images'],
                np.mean(train_stats['object_counts']) if train_stats['object_counts'] else 0,
                max(train_stats['object_counts']) if train_stats['object_counts'] else 0
            ]
            val_values = [
                val_stats['num_images'],
                np.mean(val_stats['object_counts']) if val_stats['object_counts'] else 0,
                max(val_stats['object_counts']) if val_stats['object_counts'] else 0
            ]
            
            x = np.arange(len(categories))
            width = 0.35
            
            axes[1, 2].bar(x - width/2, train_values, width, label='Training Set', alpha=0.7)
            axes[1, 2].bar(x + width/2, val_values, width, label='Validation Set', alpha=0.7)
            axes[1, 2].set_xlabel('Statistical Metrics')
            axes[1, 2].set_ylabel('Value')
            axes[1, 2].set_title('Training Set vs Validation Set Comparison')
            axes[1, 2].set_xticks(x)
            axes[1, 2].set_xticklabels(categories)
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'dataset_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Dataset analysis charts saved to: {plot_path}")
    
    def plot_training_curves(self, results_dir, save_path=None):
        """Plot training curves"""
        results_file = os.path.join(results_dir, 'results.csv')
        
        if not os.path.exists(results_file):
            print(f"Results file does not exist: {results_file}")
            return
        
        # Read training results
        import pandas as pd
        df = pd.read_csv(results_file)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('YOLO Training Curves', fontsize=16)
        
        # Loss curves
        if 'train/box_loss' in df.columns and 'val/box_loss' in df.columns:
            axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Train Box Loss', alpha=0.8)
            axes[0, 0].plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', alpha=0.8)
            axes[0, 0].set_title('Box Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Classification loss
        if 'train/cls_loss' in df.columns and 'val/cls_loss' in df.columns:
            axes[0, 1].plot(df['epoch'], df['train/cls_loss'], label='Train Cls Loss', alpha=0.8)
            axes[0, 1].plot(df['epoch'], df['val/cls_loss'], label='Val Cls Loss', alpha=0.8)
            axes[0, 1].set_title('Classification Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # mAP curves
        if 'metrics/mAP50(B)' in df.columns:
            axes[1, 0].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5', alpha=0.8)
            if 'metrics/mAP50-95(B)' in df.columns:
                axes[1, 0].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', alpha=0.8)
            axes[1, 0].set_title('mAP Metrics')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('mAP')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Precision and recall
        if 'metrics/precision(B)' in df.columns and 'metrics/recall(B)' in df.columns:
            axes[1, 1].plot(df['epoch'], df['metrics/precision(B)'], label='Precision', alpha=0.8)
            axes[1, 1].plot(df['epoch'], df['metrics/recall(B)'], label='Recall', alpha=0.8)
            axes[1, 1].set_title('Precision and Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Value')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to: {save_path}")
        
        plt.show()
    
    def compare_models(self, model_paths, data_yaml, save_path=None):
        """Compare performance of multiple models"""
        results = {}
        
        for model_path in model_paths:
            print(f"Evaluating model: {model_path}")
            try:
                model = YOLO(model_path)
                val_results = model.val(data=data_yaml, verbose=False)
                
                results[model_path] = {
                    'mAP50': val_results.box.map50,
                    'mAP50-95': val_results.box.map,
                    'precision': val_results.box.mp,
                    'recall': val_results.box.mr,
                    'f1': 2 * (val_results.box.mp * val_results.box.mr) / (val_results.box.mp + val_results.box.mr) if (val_results.box.mp + val_results.box.mr) > 0 else 0
                }
            except Exception as e:
                print(f"Error evaluating model {model_path}: {e}")
                results[model_path] = None
        
        # Plot comparison charts
        self._plot_model_comparison(results, save_path)
        
        return results
    
    def _plot_model_comparison(self, results, save_path=None):
        """Plot model comparison charts"""
        # Filter out failed models
        valid_results = {k: v for k, v in results.items() if v is not None}
        
        if not valid_results:
            print("No valid model results to compare")
            return
        
        # Extract model names and metrics
        model_names = [os.path.basename(path).replace('.pt', '') for path in valid_results.keys()]
        metrics = ['mAP50', 'mAP50-95', 'precision', 'recall', 'f1']
        
        # Create data matrix
        data_matrix = np.array([[valid_results[path][metric] for metric in metrics] for path in valid_results.keys()])
        
        # Plot heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(data_matrix, 
                   xticklabels=metrics, 
                   yticklabels=model_names,
                   annot=True, 
                   fmt='.3f', 
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Metric Value'})
        
        plt.title('Model Performance Comparison')
        plt.xlabel('Evaluation Metrics')
        plt.ylabel('Model')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Model comparison charts saved to: {save_path}")
        
        plt.show()
    
    def generate_training_report(self, results_dir, output_path='training_report.html'):
        """Generate training report HTML file"""
        # Here you can implement a more detailed HTML report generation function
        # Including training configuration, result charts, model performance, etc.
        pass


def main():
    """Example usage"""
    # Create utility instance
    utils = YOLOTrainingUtils()
    
    # Analyze dataset
    data_yaml = '/coc/pcba1/yshi457/DUCSS-DFU-Recognition/data/yolo_dataset/dataset.yaml'
    utils.load_data_config(data_yaml)
    train_stats, val_stats = utils.analyze_dataset()
    
    # Plot training curves (if training results exist)
    # utils.plot_training_curves('runs/train/exp')
    
    # Compare models
    # model_paths = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt']
    # utils.compare_models(model_paths, data_yaml)


if __name__ == "__main__":
    main()
