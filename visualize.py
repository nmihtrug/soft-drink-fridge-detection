"""
Visualization script for YOLOv11 predictions and dataset analysis.
"""

import os
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

from ultralytics import YOLO


def visualize_coco_annotations(coco_json_path: str, images_dir: str, num_samples: int = 5, output_dir: str = None):
    """
    Visualize COCO format annotations with bounding boxes.
    
    Args:
        coco_json_path: Path to _annotations.coco.json
        images_dir: Path to images directory
        num_samples: Number of sample images to visualize
        output_dir: Directory to save visualizations (optional)
    """
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Build mappings
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    img_id_to_info = {img['id']: img for img in coco_data['images']}
    
    # Group annotations by image
    img_annotations = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_annotations:
            img_annotations[img_id] = []
        img_annotations[img_id].append(ann)
    
    # Visualize samples
    sampled_imgs = coco_data['images'][:num_samples]
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 5*num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for idx, img_info in enumerate(sampled_imgs):
        img_path = os.path.join(images_dir, img_info['file_name'])
        
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        ax = axes[idx]
        ax.imshow(img)
        
        # Draw bounding boxes
        img_id = img_info['id']
        if img_id in img_annotations:
            for ann in img_annotations[img_id]:
                bbox = ann['bbox']  # [x, y, w, h]
                cat_id = ann['category_id']
                cat_name = cat_id_to_name.get(cat_id, "Unknown")
                
                x, y, w, h = bbox
                rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(x, y-5, cat_name, color='red', fontsize=8, weight='bold',
                       bbox=dict(facecolor='white', alpha=0.7, pad=1))
        
        ax.set_title(f"Image: {img_info['file_name']}")
        ax.axis('off')
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "coco_annotations.png")
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"✓ Saved visualization to {output_path}")
    
    plt.show()


def visualize_predictions(model_path: str, images_dir: str, num_samples: int = 5, conf_threshold: float = 0.5, output_dir: str = None):
    """
    Visualize YOLOv11 predictions on images.
    
    Args:
        model_path: Path to trained YOLOv11 model
        images_dir: Path to images directory
        num_samples: Number of sample images to visualize
        conf_threshold: Confidence threshold for predictions
        output_dir: Directory to save visualizations (optional)
    """
    model = YOLO(model_path)
    
    # Get image files
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    image_files = image_files[:num_samples]
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 5*num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(images_dir, img_file)
        
        # Run inference
        results = model(img_path, conf=conf_threshold, verbose=False)
        
        # Get image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        ax = axes[idx]
        ax.imshow(img)
        
        # Draw predictions
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                class_id = int(box.cls[0].item())
                class_name = result.names.get(class_id, f"Class {class_id}")
                
                rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='g', facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1-5, f"{class_name} ({conf:.2f})", color='green', fontsize=8, weight='bold',
                       bbox=dict(facecolor='white', alpha=0.7, pad=1))
        
        ax.set_title(f"Predictions: {img_file}")
        ax.axis('off')
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "predictions.png")
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"✓ Saved visualization to {output_path}")
    
    plt.show()


def plot_dataset_statistics(coco_json_path: str, output_dir: str = None):
    """
    Plot dataset statistics including class distribution and instance counts.
    
    Args:
        coco_json_path: Path to _annotations.coco.json
        output_dir: Directory to save plots (optional)
    """
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Build mappings
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Count instances per category
    cat_counts = {}
    for cat in coco_data['categories']:
        cat_counts[cat['id']] = 0
    
    for ann in coco_data['annotations']:
        cat_id = ann['category_id']
        if cat_id in cat_counts:
            cat_counts[cat_id] += 1
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Class distribution
    cat_names = [cat_id_to_name.get(cid, f"Class {cid}") for cid in sorted(cat_counts.keys())]
    counts = [cat_counts[cid] for cid in sorted(cat_counts.keys())]
    
    axes[0, 0].barh(cat_names, counts, color='skyblue')
    axes[0, 0].set_xlabel('Number of Instances')
    axes[0, 0].set_title('Class Distribution')
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    # 2. Images per class
    img_per_class = {cid: set() for cid in cat_counts.keys()}
    for ann in coco_data['annotations']:
        img_per_class[ann['category_id']].add(ann['image_id'])
    
    img_counts = [len(img_per_class[cid]) for cid in sorted(cat_counts.keys())]
    axes[0, 1].barh(cat_names, img_counts, color='lightcoral')
    axes[0, 1].set_xlabel('Number of Images')
    axes[0, 1].set_title('Images per Class')
    axes[0, 1].grid(axis='x', alpha=0.3)
    
    # 3. Statistics
    axes[1, 0].axis('off')
    stats_text = f"""
Dataset Statistics:
- Total Images: {len(coco_data['images'])}
- Total Annotations: {len(coco_data['annotations'])}
- Number of Classes: {len(coco_data['categories'])}
- Avg Instances per Image: {len(coco_data['annotations']) / len(coco_data['images']):.2f}
- Avg Instances per Class: {np.mean(counts):.2f}
"""
    axes[1, 0].text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
                    family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 4. Instance size distribution
    bbox_areas = []
    for ann in coco_data['annotations']:
        bbox = ann['bbox']
        area = bbox[2] * bbox[3]  # w * h
        bbox_areas.append(area)
    
    axes[1, 1].hist(bbox_areas, bins=30, color='lightgreen', edgecolor='black')
    axes[1, 1].set_xlabel('Bbox Area (pixels)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Bounding Box Size Distribution')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "dataset_statistics.png")
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"✓ Saved statistics to {output_path}")
    
    plt.show()


def plot_training_results(results_dir: str):
    """
    Plot training results (loss, accuracy, etc).
    
    Args:
        results_dir: Path to results directory from training
    """
    # Try to find results.csv
    results_csv = os.path.join(results_dir, "results.csv")
    
    if not os.path.exists(results_csv):
        print(f"Results file not found: {results_csv}")
        return
    
    import pandas as pd
    df = pd.read_csv(results_csv)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot loss curves
    if 'train/loss' in df.columns:
        axes[0, 0].plot(df['train/loss'], label='Train Loss', color='blue')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].grid(alpha=0.3)
        axes[0, 0].legend()
    
    # Plot validation metrics
    if 'metrics/mAP50' in df.columns:
        axes[0, 1].plot(df['metrics/mAP50'], label='mAP50', color='green')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('mAP50')
        axes[0, 1].set_title('Validation mAP50')
        axes[0, 1].grid(alpha=0.3)
        axes[0, 1].legend()
    
    # Plot precision/recall
    if 'metrics/precision' in df.columns:
        axes[1, 0].plot(df['metrics/precision'], label='Precision', color='orange')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Validation Precision')
        axes[1, 0].grid(alpha=0.3)
        axes[1, 0].legend()
    
    if 'metrics/recall' in df.columns:
        axes[1, 1].plot(df['metrics/recall'], label='Recall', color='red')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].set_title('Validation Recall')
        axes[1, 1].grid(alpha=0.3)
        axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize YOLOv11 results")
    parser.add_argument("--task", choices=["annotations", "predictions", "stats", "training"], 
                       default="annotations", help="Visualization task")
    parser.add_argument("--coco-json", help="Path to COCO annotations JSON")
    parser.add_argument("--images-dir", help="Path to images directory")
    parser.add_argument("--model", help="Path to trained YOLOv11 model")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to visualize")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--output-dir", help="Directory to save visualizations")
    parser.add_argument("--results-dir", help="Path to training results directory")
    
    args = parser.parse_args()
    
    if args.task == "annotations":
        if not args.coco_json or not args.images_dir:
            print("Error: --coco-json and --images-dir required for annotations task")
            return
        visualize_coco_annotations(args.coco_json, args.images_dir, args.num_samples, args.output_dir)
    
    elif args.task == "predictions":
        if not args.model or not args.images_dir:
            print("Error: --model and --images-dir required for predictions task")
            return
        visualize_predictions(args.model, args.images_dir, args.num_samples, args.conf, args.output_dir)
    
    elif args.task == "stats":
        if not args.coco_json:
            print("Error: --coco-json required for stats task")
            return
        plot_dataset_statistics(args.coco_json, args.output_dir)
    
    elif args.task == "training":
        if not args.results_dir:
            print("Error: --results-dir required for training task")
            return
        plot_training_results(args.results_dir)


if __name__ == "__main__":
    main()
