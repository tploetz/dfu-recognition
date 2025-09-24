import argparse
import os
import yaml
import torch
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def load_config(config_path):
    """Load training configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def validate_dataset(data_yaml):
    """Validate dataset configuration"""
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Dataset configuration file does not exist: {data_yaml}")
    
    with open(data_yaml, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    
    # Check paths
    base_path = data_config.get('path', '')
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Dataset base path does not exist: {base_path}")
    
    # Check training and validation sets
    train_path = os.path.join(base_path, data_config.get('train', ''))
    val_path = os.path.join(base_path, data_config.get('val', ''))
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training set path does not exist: {train_path}")
    
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation set path does not exist: {val_path}")
    
    print(f"✓ Dataset validation passed")
    print(f"  - Base path: {base_path}")
    print(f"  - Training set: {train_path}")
    print(f"  - Validation set: {val_path}")
    print(f"  - Number of classes: {data_config.get('nc', 0)}")
    print(f"  - Class names: {data_config.get('names', [])}")
    
    return data_config


def setup_training_args(args):
    """Setup training arguments"""
    training_args = {}
    
    # Basic training parameters
    if hasattr(args, 'data') and args.data:
        training_args['data'] = args.data
    if hasattr(args, 'epochs') and args.epochs:
        training_args['epochs'] = args.epochs
    if hasattr(args, 'imgsz') and args.imgsz:
        training_args['imgsz'] = args.imgsz
    if hasattr(args, 'batch') and args.batch:
        training_args['batch'] = args.batch
    if hasattr(args, 'device') and args.device:
        training_args['device'] = args.device
    if hasattr(args, 'workers') and args.workers:
        training_args['workers'] = args.workers
    if hasattr(args, 'project') and args.project:
        training_args['project'] = args.project
    if hasattr(args, 'name') and args.name:
        training_args['name'] = args.name
    if hasattr(args, 'exist_ok') and args.exist_ok:
        training_args['exist_ok'] = args.exist_ok
    
    # Training parameters
    if hasattr(args, 'pretrained') and args.pretrained:
        training_args['pretrained'] = args.pretrained
    if hasattr(args, 'optimizer') and args.optimizer:
        training_args['optimizer'] = args.optimizer
    if hasattr(args, 'verbose') and args.verbose:
        training_args['verbose'] = args.verbose
    if hasattr(args, 'seed') and args.seed is not None:
        training_args['seed'] = args.seed
    if hasattr(args, 'deterministic') and args.deterministic:
        training_args['deterministic'] = args.deterministic
    if hasattr(args, 'single_cls') and args.single_cls:
        training_args['single_cls'] = args.single_cls
    if hasattr(args, 'rect') and args.rect:
        training_args['rect'] = args.rect
    if hasattr(args, 'cos_lr') and args.cos_lr:
        training_args['cos_lr'] = args.cos_lr
    if hasattr(args, 'close_mosaic') and args.close_mosaic:
        training_args['close_mosaic'] = args.close_mosaic
    if hasattr(args, 'resume') and args.resume:
        training_args['resume'] = args.resume
    if hasattr(args, 'amp') and args.amp:
        training_args['amp'] = args.amp
    if hasattr(args, 'fraction') and args.fraction:
        training_args['fraction'] = args.fraction
    if hasattr(args, 'profile') and args.profile:
        training_args['profile'] = args.profile
    if hasattr(args, 'freeze') and args.freeze:
        training_args['freeze'] = args.freeze
    
    # Learning rate parameters
    if hasattr(args, 'lr0') and args.lr0:
        training_args['lr0'] = args.lr0
    if hasattr(args, 'lrf') and args.lrf:
        training_args['lrf'] = args.lrf
    if hasattr(args, 'momentum') and args.momentum:
        training_args['momentum'] = args.momentum
    if hasattr(args, 'weight_decay') and args.weight_decay:
        training_args['weight_decay'] = args.weight_decay
    if hasattr(args, 'warmup_epochs') and args.warmup_epochs:
        training_args['warmup_epochs'] = args.warmup_epochs
    if hasattr(args, 'warmup_momentum') and args.warmup_momentum:
        training_args['warmup_momentum'] = args.warmup_momentum
    if hasattr(args, 'warmup_bias_lr') and args.warmup_bias_lr:
        training_args['warmup_bias_lr'] = args.warmup_bias_lr
    
    # Loss function parameters
    if hasattr(args, 'box') and args.box:
        training_args['box'] = args.box
    if hasattr(args, 'cls') and args.cls:
        training_args['cls'] = args.cls
    if hasattr(args, 'dfl') and args.dfl:
        training_args['dfl'] = args.dfl
    if hasattr(args, 'pose') and args.pose:
        training_args['pose'] = args.pose
    if hasattr(args, 'kobj') and args.kobj:
        training_args['kobj'] = args.kobj
    if hasattr(args, 'label_smoothing') and args.label_smoothing:
        training_args['label_smoothing'] = args.label_smoothing
    if hasattr(args, 'nbs') and args.nbs:
        training_args['nbs'] = args.nbs
    if hasattr(args, 'overlap_mask') and args.overlap_mask:
        training_args['overlap_mask'] = args.overlap_mask
    if hasattr(args, 'mask_ratio') and args.mask_ratio:
        training_args['mask_ratio'] = args.mask_ratio
    if hasattr(args, 'dropout') and args.dropout:
        training_args['dropout'] = args.dropout
    
    # Validation parameters
    if hasattr(args, 'val') and args.val:
        training_args['val'] = args.val
    if hasattr(args, 'split') and args.split:
        training_args['split'] = args.split
    if hasattr(args, 'save_json') and args.save_json:
        training_args['save_json'] = args.save_json
    if hasattr(args, 'save_hybrid') and args.save_hybrid:
        training_args['save_hybrid'] = args.save_hybrid
    if hasattr(args, 'conf') and args.conf:
        training_args['conf'] = args.conf
    if hasattr(args, 'iou') and args.iou:
        training_args['iou'] = args.iou
    if hasattr(args, 'max_det') and args.max_det:
        training_args['max_det'] = args.max_det
    if hasattr(args, 'half') and args.half:
        training_args['half'] = args.half
    if hasattr(args, 'dnn') and args.dnn:
        training_args['dnn'] = args.dnn
    if hasattr(args, 'plots') and args.plots:
        training_args['plots'] = args.plots
    
    # Data augmentation parameters
    if hasattr(args, 'hsv_h') and args.hsv_h:
        training_args['hsv_h'] = args.hsv_h
    if hasattr(args, 'hsv_s') and args.hsv_s:
        training_args['hsv_s'] = args.hsv_s
    if hasattr(args, 'hsv_v') and args.hsv_v:
        training_args['hsv_v'] = args.hsv_v
    if hasattr(args, 'degrees') and args.degrees:
        training_args['degrees'] = args.degrees
    if hasattr(args, 'translate') and args.translate:
        training_args['translate'] = args.translate
    if hasattr(args, 'scale') and args.scale:
        training_args['scale'] = args.scale
    if hasattr(args, 'shear') and args.shear:
        training_args['shear'] = args.shear
    if hasattr(args, 'perspective') and args.perspective:
        training_args['perspective'] = args.perspective
    if hasattr(args, 'flipud') and args.flipud:
        training_args['flipud'] = args.flipud
    if hasattr(args, 'fliplr') and args.fliplr:
        training_args['fliplr'] = args.fliplr
    if hasattr(args, 'mosaic') and args.mosaic:
        training_args['mosaic'] = args.mosaic
    if hasattr(args, 'mixup') and args.mixup:
        training_args['mixup'] = args.mixup
    if hasattr(args, 'copy_paste') and args.copy_paste:
        training_args['copy_paste'] = args.copy_paste
    
    return training_args


def setup_model_path(model_name, models_dir='models'):
    """Setup model path and download if necessary"""
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Check if it's a pretrained model name
    pretrained_models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt',
                        'yolov9t.pt', 'yolov9s.pt', 'yolov9m.pt', 'yolov9c.pt', 'yolov9e.pt',
                        'yolov10n.pt', 'yolov10s.pt', 'yolov10m.pt', 'yolov10b.pt', 'yolov10l.pt', 'yolov10x.pt',
                        'yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt']
    
    if model_name in pretrained_models:
        # For pretrained models, save to models directory
        model_path = os.path.join(models_dir, model_name)
        if not os.path.exists(model_path):
            print(f"Downloading pretrained model: {model_name}")
            # YOLO will automatically download and save to the specified path
            model = YOLO(model_name)
            # Save the model to our models directory
            model.save(model_path)
            print(f"Model saved to: {model_path}")
        else:
            print(f"Using existing model: {model_path}")
    else:
        # For custom model paths, use as is
        model_path = model_name
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        print(f"Using custom model: {model_path}")
    
    return model_path


def train_model(model_path, data_yaml, training_args):
    """Train YOLO model"""
    print(f"Starting model training: {model_path}")
    print(f"Using dataset: {data_yaml}")
    
    # Load model
    model = YOLO(model_path)
    
    # Start training
    results = model.train(**training_args)
    
    return results, model


def validate_model(model, data_yaml, **kwargs):
    """Validate model"""
    print("Starting model validation...")
    results = model.val(data=data_yaml, **kwargs)
    return results


def export_model(model, format='onnx', **kwargs):
    """Export model"""
    print(f"Exporting model to {format} format...")
    exported_model = model.export(format=format, **kwargs)
    return exported_model


def plot_training_results(results, save_path=None):
    """Plot training results"""
    try:
        # Get training history
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
        else:
            print("Unable to get training results, skipping plot")
            return
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('YOLO Training Results', fontsize=16)
        
        # Loss curves
        if 'train/box_loss' in metrics and 'val/box_loss' in metrics:
            axes[0, 0].plot(metrics['train/box_loss'], label='Train Box Loss')
            axes[0, 0].plot(metrics['val/box_loss'], label='Val Box Loss')
            axes[0, 0].set_title('Box Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # Classification loss
        if 'train/cls_loss' in metrics and 'val/cls_loss' in metrics:
            axes[0, 1].plot(metrics['train/cls_loss'], label='Train Cls Loss')
            axes[0, 1].plot(metrics['val/cls_loss'], label='Val Cls Loss')
            axes[0, 1].set_title('Classification Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # mAP curves
        if 'metrics/mAP50' in metrics:
            axes[1, 0].plot(metrics['metrics/mAP50'], label='mAP@0.5')
            if 'metrics/mAP50-95' in metrics:
                axes[1, 0].plot(metrics['metrics/mAP50-95'], label='mAP@0.5:0.95')
            axes[1, 0].set_title('mAP Metrics')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('mAP')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Precision and recall
        if 'metrics/precision' in metrics and 'metrics/recall' in metrics:
            axes[1, 1].plot(metrics['metrics/precision'], label='Precision')
            axes[1, 1].plot(metrics['metrics/recall'], label='Recall')
            axes[1, 1].set_title('Precision and Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Value')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training results plot saved to: {save_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"Error plotting training results: {e}")


def main():
    parser = argparse.ArgumentParser(description='YOLO Model Training Tool')
    
    # Basic parameters
    parser.add_argument('--model', '-m', type=str, default='yolov8n.pt',
                       help='Model file path or pretrained model name (default: yolov8n.pt)')
    parser.add_argument('--data', '-d', type=str, required=True,
                       help='Dataset configuration file path (YAML format)')
    parser.add_argument('--epochs', '-e', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--imgsz', '-s', type=int, default=640,
                       help='Input image size (default: 640)')
    parser.add_argument('--batch', '-b', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--device', type=str, default='',
                       help='Training device (default: auto select)')
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of data loader worker processes (default: 8)')
    
    # Output parameters
    parser.add_argument('--project', type=str, default='models/runs/train',
                       help='Project save path (default: models/runs/train)')
    parser.add_argument('--name', type=str, default='exp',
                       help='Experiment name (default: exp)')
    parser.add_argument('--exist_ok', action='store_true',
                       help='Allow overwriting existing experiments')
    parser.add_argument('--models_dir', type=str, default='models',
                       help='Directory to save downloaded models (default: models)')
    
    # Training parameters
    parser.add_argument('--pretrained', action='store_true',
                       help='Use pretrained weights')
    parser.add_argument('--optimizer', type=str, default='auto',
                       choices=['SGD', 'Adam', 'AdamW', 'RMSProp', 'auto'],
                       help='Optimizer type (default: auto)')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed (default: 0)')
    parser.add_argument('--deterministic', action='store_true',
                       help='Deterministic training')
    parser.add_argument('--single_cls', action='store_true',
                       help='Single class training')
    parser.add_argument('--rect', action='store_true',
                       help='Rectangular training')
    parser.add_argument('--cos_lr', action='store_true',
                       help='Cosine learning rate scheduler')
    parser.add_argument('--close_mosaic', type=int, default=10,
                       help='Epochs to close mosaic augmentation (default: 10)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from last checkpoint')
    parser.add_argument('--amp', action='store_true',
                       help='Automatic mixed precision training')
    parser.add_argument('--fraction', type=float, default=1.0,
                       help='Fraction of dataset to use (default: 1.0)')
    parser.add_argument('--profile', action='store_true',
                       help='Profile performance')
    parser.add_argument('--freeze', type=int, default=None,
                       help='Freeze first N layers')
    
    # Learning rate parameters
    parser.add_argument('--lr0', type=float, default=0.01,
                       help='Initial learning rate (default: 0.01)')
    parser.add_argument('--lrf', type=float, default=0.01,
                       help='Final learning rate ratio (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.937,
                       help='SGD momentum (default: 0.937)')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                       help='Weight decay (default: 0.0005)')
    parser.add_argument('--warmup_epochs', type=int, default=3,
                       help='Warmup epochs (default: 3)')
    parser.add_argument('--warmup_momentum', type=float, default=0.8,
                       help='Warmup momentum (default: 0.8)')
    parser.add_argument('--warmup_bias_lr', type=float, default=0.1,
                       help='Warmup bias learning rate (default: 0.1)')
    
    # Loss function parameters
    parser.add_argument('--box', type=float, default=7.5,
                       help='Box loss weight (default: 7.5)')
    parser.add_argument('--cls', type=float, default=0.5,
                       help='Classification loss weight (default: 0.5)')
    parser.add_argument('--dfl', type=float, default=1.5,
                       help='DFL loss weight (default: 1.5)')
    parser.add_argument('--pose', type=float, default=12.0,
                       help='Pose loss weight (default: 12.0)')
    parser.add_argument('--kobj', type=float, default=2.0,
                       help='Keypoint object loss weight (default: 2.0)')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                       help='Label smoothing (default: 0.0)')
    parser.add_argument('--nbs', type=int, default=64,
                       help='Nominal batch size (default: 64)')
    parser.add_argument('--overlap_mask', action='store_true',
                       help='Overlap mask')
    parser.add_argument('--mask_ratio', type=int, default=4,
                       help='Mask downsampling ratio (default: 4)')
    parser.add_argument('--dropout', type=float, default=0.0,
                       help='Dropout probability (default: 0.0)')
    
    # Validation parameters
    parser.add_argument('--val', action='store_true',
                       help='Validate during training')
    parser.add_argument('--split', type=str, default='val',
                       choices=['val', 'test'],
                       help='Validation set split (default: val)')
    parser.add_argument('--save_json', action='store_true',
                       help='Save COCO format JSON results')
    parser.add_argument('--save_hybrid', action='store_true',
                       help='Save hybrid labels')
    parser.add_argument('--conf', type=float, default=0.001,
                       help='Confidence threshold (default: 0.001)')
    parser.add_argument('--iou', type=float, default=0.6,
                       help='IoU threshold (default: 0.6)')
    parser.add_argument('--max_det', type=int, default=300,
                       help='Maximum detections (default: 300)')
    parser.add_argument('--half', action='store_true',
                       help='Half precision inference')
    parser.add_argument('--dnn', action='store_true',
                       help='Use OpenCV DNN')
    parser.add_argument('--plots', action='store_true',
                       help='Save training plots')
    
    # Data augmentation parameters
    parser.add_argument('--hsv_h', type=float, default=0.015,
                       help='HSV hue augmentation (default: 0.015)')
    parser.add_argument('--hsv_s', type=float, default=0.7,
                       help='HSV saturation augmentation (default: 0.7)')
    parser.add_argument('--hsv_v', type=float, default=0.4,
                       help='HSV value augmentation (default: 0.4)')
    parser.add_argument('--degrees', type=float, default=0.0,
                       help='Rotation angle range (default: 0.0)')
    parser.add_argument('--translate', type=float, default=0.1,
                       help='Translation range (default: 0.1)')
    parser.add_argument('--scale', type=float, default=0.5,
                       help='Scale range (default: 0.5)')
    parser.add_argument('--shear', type=float, default=0.0,
                       help='Shear angle range (default: 0.0)')
    parser.add_argument('--perspective', type=float, default=0.0,
                       help='Perspective transformation (default: 0.0)')
    parser.add_argument('--flipud', type=float, default=0.0,
                       help='Vertical flip probability (default: 0.0)')
    parser.add_argument('--fliplr', type=float, default=0.5,
                       help='Horizontal flip probability (default: 0.5)')
    parser.add_argument('--mosaic', type=float, default=1.0,
                       help='Mosaic augmentation probability (default: 1.0)')
    parser.add_argument('--mixup', type=float, default=0.0,
                       help='Mixup augmentation probability (default: 0.0)')
    parser.add_argument('--copy_paste', type=float, default=0.0,
                       help='Copy-paste augmentation probability (default: 0.0)')
    
    # Other parameters
    parser.add_argument('--list_models', action='store_true',
                       help='List available pretrained models')
    parser.add_argument('--validate_only', action='store_true',
                       help='Only validate model, do not train')
    parser.add_argument('--export', type=str, default=None,
                       choices=['onnx', 'torchscript', 'tflite', 'pb', 'saved_model', 'paddle', 'ncnn', 'openvino', 'coreml'],
                       help='Export model format')
    parser.add_argument('--plot_results', action='store_true',
                       help='Plot training results')
    
    args = parser.parse_args()
    
    # List available models
    if args.list_models:
        print("Available pretrained YOLO models:")
        print("YOLOv8 models:")
        print("  - yolov8n.pt (nano - fastest)")
        print("  - yolov8s.pt (small)")
        print("  - yolov8m.pt (medium)")
        print("  - yolov8l.pt (large)")
        print("  - yolov8x.pt (extra large - most accurate)")
        print("\nYOLOv9 models:")
        print("  - yolov9t.pt (tiny)")
        print("  - yolov9s.pt (small)")
        print("  - yolov9m.pt (medium)")
        print("  - yolov9c.pt (classic)")
        print("  - yolov9e.pt (extra)")
        print("\nYOLOv10 models:")
        print("  - yolov10n.pt (nano)")
        print("  - yolov10s.pt (small)")
        print("  - yolov10m.pt (medium)")
        print("  - yolov10b.pt (base)")
        print("  - yolov10l.pt (large)")
        print("  - yolov10x.pt (extra large)")
        print("\nNote: Models will be automatically downloaded if not found locally.")
        return
    
    try:
        # Validate dataset
        print("Validating dataset configuration...")
        data_config = validate_dataset(args.data)
        
        # Setup model path and download if necessary
        print(f"Setting up model: {args.model}")
        model_path = setup_model_path(args.model, args.models_dir)
        
        # Setup training arguments
        training_args = setup_training_args(args)
        
        # Load model
        print(f"Loading model: {model_path}")
        model = YOLO(model_path)
        
        if args.validate_only:
            # Validation only mode
            print("Starting model validation...")
            results = validate_model(model, args.data, **training_args)
            print("Validation completed!")
            return
        
        # Start training
        print("Starting training...")
        results = train_model(model_path, args.data, training_args)
        
        print("Training completed!")
        print(f"Results saved in: {args.project}/{args.name}")
        
        # Plot training results
        if args.plot_results:
            plot_save_path = os.path.join(args.project, args.name, 'training_results.png')
            plot_training_results(results, plot_save_path)
        
        # Export model
        if args.export:
            export_path = os.path.join(args.models_dir, f"exported_{args.name}.{args.export}")
            export_model(model, format=args.export)
            print(f"Model exported to {args.export} format")
        
        # Final validation
        print("Performing final validation...")
        final_results = validate_model(model, args.data)
        
        print("\nTraining summary:")
        print(f"- Epochs: {args.epochs}")
        print(f"- Batch size: {args.batch}")
        print(f"- Image size: {args.imgsz}")
        print(f"- Dataset: {args.data}")
        print(f"- Model: {model_path}")
        print(f"- Models directory: {args.models_dir}")
        print(f"- Results directory: {args.project}/{args.name}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()