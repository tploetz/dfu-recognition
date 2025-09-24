# DFU Detection Pipeline

A comprehensive diabetes foot ulcer (DFU) detection pipeline that combines YOLO object detection and SAM2-UNet precise segmentation for accurate lesion identification and analysis.

## Features

- 🎯 **Dual Detection**: Combines YOLO object detection and SAM2-UNet precise segmentation
- 📁 **Batch Processing**: Supports folder-based batch detection
- 🖼️ **Single Image**: Supports single image detection
- 🔧 **Direct Function Calls**: High-performance direct function calls without subprocess overhead
- 💾 **Result Integration**: Automatically combines detection results from both methods
- ⚙️ **Configurable Parameters**: Supports confidence thresholds, device selection, and other parameters
- 🔧 **Command Line Interface**: Complete command-line interface
- 📊 **Progress Tracking**: Real-time progress bars using tqdm
- 🛠️ **Device Compatibility**: Intelligent handling of model loading across different devices
- 📈 **Evaluation Tools**: Built-in evaluation metrics with progress tracking

## Environment Setup

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU acceleration)
- 8GB+ RAM (16GB+ recommended for large datasets)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd DUCSS-DFU-Recognition
   ```

2. **Create and activate virtual environment**:
   ```bash
   # Using conda (recommended)
   conda create -n dfu python=3.10
   conda activate dfu
   
   # Or using venv
   python -m venv dfu_env
   source dfu_env/bin/activate  # Linux/Mac
   # or
   dfu_env\Scripts\activate  # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**:
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}')"
   python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
   python -c "import ultralytics; print('Ultralytics: OK')"
   ```

### Model Files

Ensure the following model files are available:
- `models/yolo.pt` - YOLO model
- `models/SAM2-UNet.pth` - SAM2-UNet model

## Detection Pipeline

### Step 1: YOLO Object Detection
- Uses trained YOLO model for initial target detection
- Outputs bounding box coordinates and confidence scores
- Saves YOLO format txt label files

### Step 2: SAM2-UNet Precise Segmentation
- Uses SAM2-UNet model for precise pixel-level segmentation
- Outputs high-precision segmentation masks
- Maintains original input image dimensions

### Step 3: Result Integration
- Draws bounding box masks from YOLO detection results
- Calculates intersection with SAM segmentation masks
- Generates final identification images and visualization results

## Usage

### Python Function Calls (Recommended)

#### Complete Pipeline
```python
from dfu import DFUDetector

# Initialize detector
detector = DFUDetector(
    yolo_model_path="models/yolo.pt",
    sam_model_path="models/SAM2-UNet.pth",
    device="0"
)

# Run complete detection pipeline
results = detector.run_full_detection(
    input_path="data/test_images",
    output_base_dir="prediction/dfu_results",
    yolo_conf=0.25,
    save_yolo_txt=True,
    use_sam_dataset=True
)

print(f"Processed {results['steps']['final']['processed_images']} images")
```

#### Step-by-Step Execution
```python
# 1. YOLO Detection
yolo_result = detector.run_yolo_detection(
    input_path="data/test_images",
    output_dir="prediction/yolo_results",
    conf_threshold=0.25
)

# 2. SAM Segmentation
sam_result = detector.run_sam_detection(
    input_path="data/test_images",
    output_dir="prediction/sam_results",
    use_dataset=True
)

# 3. Generate Final Results
final_result = detector.generate_final_results(
    input_path="data/test_images",
    yolo_output_dir="prediction/yolo_results",
    sam_output_dir="prediction/sam_results",
    final_output_dir="prediction/final_results"
)
```

### Command Line Interface

#### Basic Usage
```bash
python dfu.py \
    --yolo_model models/yolo.pt \
    --sam_model models/SAM2-UNet.pth \
    --input data/test_images \
    --output prediction/dfu_results \
    --device 0 \
    --yolo_conf 0.25
```

#### Single Image Detection
```bash
python dfu.py \
    --yolo_model models/yolo.pt \
    --sam_model models/SAM2-UNet.pth \
    --input data/test_image.png \
    --output prediction/single_result \
    --device 0
```

#### Individual Tool Usage

**YOLO Detection**:
```bash
python detect_yolo.py \
    --model models/yolo.pt \
    --input data/test_images \
    --output prediction/yolo_results \
    --device 0 \
    --conf 0.25 \
    --save_yolo_txt
```

**SAM2-UNet Segmentation**:
```bash
python detect_sam.py \
    --checkpoint models/SAM2-UNet.pth \
    --input data/test_images \
    --output prediction/sam_results \
    --cuda_idx 0 \
    --use_dataset
```

## Output Structure

```
output_base_dir/
├── yolo_results/           # YOLO detection results
│   ├── image1.txt          # YOLO format labels
│   ├── image1_detection.json # Detailed detection results
│   └── ...
├── sam_results/            # SAM2-UNet segmentation results
│   ├── image1_sam_mask.png # Segmentation masks
│   └── ...
├── combined_results/       # Final identification results (final masks only)
│   ├── image1_final_mask.png   # Final identification mask
│   └── ...
├── visualization/          # Visualization results
│   ├── image1_yolo_visual.png  # YOLO visualization
│   ├── image1_sam_visual.png   # SAM visualization
│   └── image1_overlap_visual.png # Intersection visualization
└── yolo_bbox/             # YOLO bounding box masks
    ├── image1_yolo_bbox.png    # YOLO bounding box mask
    └── ...
```

## File Formats

### YOLO Detection Results (txt format)
```
class_id center_x center_y width height
0 0.5 0.5 0.3 0.4
```

### SAM Segmentation Masks (PNG format)
- 8-bit grayscale images
- Value range: 0-255
- Same dimensions as input images

### Final Results (PNG format)
- **Final Identification Mask**: `*_final_mask.png` - Intersection of YOLO bounding boxes and SAM segmentation (final result)
- **YOLO Bounding Box Mask**: `*_yolo_bbox.png` - Bounding box masks drawn from YOLO detection results
- **Visualization Results**:
  - `*_yolo_visual.png` - YOLO detection visualization
  - `*_sam_visual.png` - SAM segmentation visualization
  - `*_overlap_visual.png` - Intersection visualization

## Evaluation

### Run Evaluation
```bash
python eval.py \
    --pred_path prediction/dfu/combined_results \
    --gt_path data/DFU_split/ulcer/test/labels \
    --dataset_name DFU
```

### Evaluation Metrics
- **mDice**: Mean Dice coefficient
- **mIoU**: Mean Intersection over Union
- **S_α**: S-measure
- **F^w_β**: Weighted F-measure
- **F_β**: Adaptive F-measure
- **E_φ**: Mean E-measure
- **MAE**: Mean Absolute Error


## Troubleshooting

### Common Issues

1. **Model Loading Failed**:
   - Check model file paths and formats
   - Verify CUDA compatibility
   - Ensure sufficient GPU memory

2. **Device Errors**:
   - Check CUDA availability: `torch.cuda.is_available()`
   - Verify device indices: `torch.cuda.device_count()`
   - Use CPU if GPU unavailable: `--device cpu`

3. **Memory Issues**:
   - Reduce batch size
   - Use CPU processing
   - Process smaller image batches

4. **File Permission Errors**:
   - Check read/write permissions for input/output directories
   - Ensure sufficient disk space

5. **Import Errors**:
   - Verify virtual environment activation
   - Reinstall dependencies: `pip install -r requirements.txt`
   - Check Python version compatibility

### Debug Mode

Enable detailed logging:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
python dfu.py --help
```

## License

This project is licensed under the MIT License.

## Citation

If you use this code in your research, please cite:

```bibtex
@{,
  title={},
  author={},
  year={},
  url={}
}
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

## Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the documentation

---

**Note**: This pipeline is designed for research and educational purposes. Please ensure proper validation before clinical use.