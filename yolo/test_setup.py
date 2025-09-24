#!/usr/bin/env python3
"""
Test if YOLO module setup is correct
"""

import sys
import os

def test_imports():
    """Test if imports are working correctly"""
    print("Testing imports...")
    
    try:
        # Test ultralytics import
        from ultralytics import YOLO
        print("✓ ultralytics import successful")
    except ImportError as e:
        print(f"✗ ultralytics import failed: {e}")
        return False
    
    try:
        # Test utility class import
        from utils import YOLOTrainingUtils
        print("✓ YOLOTrainingUtils import successful")
    except ImportError as e:
        print(f"✗ YOLOTrainingUtils import failed: {e}")
        return False
    
    return True

def test_file_structure():
    """Test file structure"""
    print("\nTesting file structure...")
    
    required_files = [
        '../train_yolo.py',
        'utils/train_utils.py',
        'utils/__init__.py',
        'examples/example_usage.py',
        'examples/__init__.py',
        'configs/training_config.yaml',
        'configs/__init__.py',
        'requirements.txt',
        'README.md',
        'README_training.md'
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} does not exist")
            all_exist = False
    
    return all_exist

def test_data_paths():
    """Test data paths"""
    print("\nTesting data paths...")
    
    data_yaml = '../data/yolo_dataset/dataset.yaml'
    if os.path.exists(data_yaml):
        print(f"✓ Dataset configuration file exists: {data_yaml}")
        return True
    else:
        print(f"✗ Dataset configuration file does not exist: {data_yaml}")
        return False

def main():
    """Main function"""
    print("YOLO Module Setup Test")
    print("=" * 40)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test file structure
    structure_ok = test_file_structure()
    
    # Test data paths
    data_ok = test_data_paths()
    
    print("\n" + "=" * 40)
    print("Test Results:")
    print(f"Import test: {'Passed' if imports_ok else 'Failed'}")
    print(f"File structure: {'Passed' if structure_ok else 'Failed'}")
    print(f"Data paths: {'Passed' if data_ok else 'Failed'}")
    
    if imports_ok and structure_ok:
        print("\n✓ YOLO module setup is correct!")
        print("\nYou can start training with the following command:")
        print("python ../train_yolo.py --model yolov8n.pt --data ../data/yolo_dataset/dataset.yaml --epochs 50")
    else:
        print("\n✗ YOLO module setup has issues, please check the above errors")
    
    return imports_ok and structure_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
