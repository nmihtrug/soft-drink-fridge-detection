"""
Training script for YOLOv11 on Soft Drinks Fridge dataset with synthetic data augmentation.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
from ultralytics import YOLO
import yaml

from synthetic_data import generate_synthetic_dataset


def setup_dataset_yaml(data_root: str, output_path: str = "data.yaml"):
    """
    Create a YAML file for YOLOv11 training with COCO format data.
    
    Args:
        data_root: Path to dataset root containing train/valid/test folders
        output_path: Path to save the YAML config
    """
    dataset_config = {
        "path": os.path.abspath(data_root),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 7,  # Number of classes (soft drink categories)
        "names": {
            0: "CocaCola",
            1: "Fanta",
            2: "Gingerbeer",
            3: "Portello",
            4: "Soda",
            5: "Sprite",
            6: "Water"
        }
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"✓ Created dataset config: {output_path}")
    return output_path


def convert_coco_to_yolo(coco_json_path: str, images_dir: str, output_dir: str):
    """
    Convert COCO format annotations to YOLO format (one txt file per image).
    
    Args:
        coco_json_path: Path to _annotations.coco.json
        images_dir: Path to directory containing images
        output_dir: Path to save YOLO format annotations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Build category mapping
    cat_id_to_idx = {}
    for cat in coco_data.get('categories', []):
        cat_id_to_idx[cat['id']] = cat['id']  # Using category_id directly as class index
    
    # Build image_id to annotations mapping
    img_annotations = {}
    for img in coco_data['images']:
        img_annotations[img['id']] = {
            'file_name': img['file_name'],
            'width': img['width'],
            'height': img['height'],
            'annotations': []
        }
    
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id in img_annotations:
            img_annotations[img_id]['annotations'].append(ann)
    
    # Convert to YOLO format
    converted_count = 0
    for img_id, img_data in img_annotations.items():
        if not img_data['annotations']:
            continue
        
        # Create txt file with same name as image
        base_name = Path(img_data['file_name']).stem
        txt_path = os.path.join(output_dir, f"{base_name}.txt")
        
        W, H = img_data['width'], img_data['height']
        
        with open(txt_path, 'w') as f:
            for ann in img_data['annotations']:
                bbox = ann['bbox']  # [x, y, w, h]
                cat_id = ann['category_id']
                
                if cat_id not in cat_id_to_idx:
                    continue
                
                class_idx = cat_id_to_idx[cat_id]
                
                # Convert to YOLO format: [class_id, x_center, y_center, width, height] (normalized)
                x, y, w, h = bbox
                x_center = (x + w / 2) / W
                y_center = (y + h / 2) / H
                w_norm = w / W
                h_norm = h / H
                
                f.write(f"{class_idx} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
        
        converted_count += 1
    
    return converted_count


def prepare_data_for_yolo(data_root: str):
    """
    Prepare dataset for YOLOv11 by converting COCO format to YOLO format.
    Creates images/ and labels/ directories in each split folder.
    
    Args:
        data_root: Path to dataset root
    """
    print("Converting COCO annotations to YOLO format...")
    
    for split in ['train', 'valid', 'test']:
        split_dir = os.path.join(data_root, split)
        
        if not os.path.exists(split_dir):
            print(f"⚠ Skipping {split}: directory not found")
            continue
        
        # Create images and labels directories
        images_dir = os.path.join(split_dir, 'images')
        labels_dir = os.path.join(split_dir, 'labels')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        # Move existing images to images/ directory
        for file in os.listdir(split_dir):
            if file.endswith(('.jpg', '.jpeg', '.png')):
                src = os.path.join(split_dir, file)
                dst = os.path.join(images_dir, file)
                if not os.path.exists(dst):
                    os.rename(src, dst)
        
        # Convert COCO to YOLO format
        coco_json = os.path.join(split_dir, '_annotations.coco.json')
        if os.path.exists(coco_json):
            count = convert_coco_to_yolo(coco_json, images_dir, labels_dir)
            print(f"  ✓ {split}: converted {count} images")


def train_yolov11(
    data_yaml: str,
    model_name: str = "yolov11n",
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    device: int = 0,
    patience: int = 20,
    output_dir: str = "runs/train"
):
    """
    Train YOLOv11 model.

    Args:
        data_yaml: Path to dataset YAML config
        model_name: YOLOv11 model variant (n, s, m, l, x)
        epochs: Number of training epochs
        batch_size: Batch size
        img_size: Image size
        device: GPU device ID
        patience: Early stopping patience
        output_dir: Directory to save results
    """
    # Load model
    model = YOLO(f"{model_name}.pt")

    # Train
    print(f"\n🚀 Starting training with {model_name}...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        patience=patience,
        project=output_dir,
        name=f"yolov11_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        save=True,
        cache=False,
        plots=True,
        mosaic=1.0,
        flipud=0.5,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        perspective=0.0,
        verbose=True
    )

    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv11 on Soft Drinks Fridge dataset")
    parser.add_argument("--data-root", default="data/synthetic", help="Path to dataset root")
    parser.add_argument("--generate-synthetic", action="store_true", help="Generate synthetic data before training")
    parser.add_argument("--input-data-root", default="data/original", help="Path to original data (for synthetic generation)")
    parser.add_argument("--model", default="yolov11n", choices=["yolov11n", "yolov11s", "yolov11m", "yolov11l", "yolov11x"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--no-prepare", action="store_true", help="Skip YOLO format preparation")
    
    args = parser.parse_args()
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"✓ GPU available: {torch.cuda.get_device_name(args.device)}")
    else:
        print("⚠ GPU not available, using CPU")
    
    # Generate synthetic data if requested
    if args.generate_synthetic:
        print("\n📊 Generating synthetic data...")
        generate_synthetic_dataset(
            input_root=args.input_data_root,
            output_root=args.data_root,
            num_aug_per_image=20
        )
    
    # Prepare data for YOLO format
    if not args.no_prepare:
        print("\n🔄 Preparing data for YOLOv11...")
        prepare_data_for_yolo(args.data_root)
    
    # Setup dataset YAML
    data_yaml = setup_dataset_yaml(args.data_root)
    
    # Train
    results = train_yolov11(
        data_yaml=data_yaml,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        device=args.device,
        patience=args.patience
    )
    
    print("\n✅ Training completed!")
    print(f"Results saved in: runs/train")


if __name__ == "__main__":
    main()
