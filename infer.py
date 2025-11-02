"""
Simple inference script for running YOLOv11 predictions on images or video.
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def run_image_inference(model_path: str, image_path: str, conf: float = 0.5, save: bool = False):
    """
    Run inference on a single image.
    
    Args:
        model_path: Path to trained model
        image_path: Path to image
        conf: Confidence threshold
        save: Save output image
    """
    model = YOLO(model_path)
    
    # Run inference
    results = model(image_path, conf=conf, verbose=False)
    
    if results and len(results) > 0:
        result = results[0]
        
        # Print detections
        print(f"\n📸 Image: {image_path}")
        print(f"Detections: {len(result.boxes)}")
        print("-" * 50)
        
        for i, box in enumerate(result.boxes):
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            conf_score = box.conf[0].item()
            class_id = int(box.cls[0].item())
            class_name = result.names.get(class_id, f"Class {class_id}")
            
            print(f"{i+1}. {class_name}")
            print(f"   Confidence: {conf_score:.2%}")
            print(f"   Bbox: ({x1:.1f}, {y1:.1f}) -> ({x2:.1f}, {y2:.1f})")
        
        # Save result
        if save:
            output_path = f"detection_{Path(image_path).stem}.jpg"
            result.save(output_path)
            print(f"\n✓ Saved to {output_path}")
    else:
        print(f"No detections found in {image_path}")


def run_batch_inference(model_path: str, image_dir: str, conf: float = 0.5, output_dir: str = None):
    """
    Run inference on all images in a directory.
    
    Args:
        model_path: Path to trained model
        image_dir: Path to directory with images
        conf: Confidence threshold
        output_dir: Directory to save results (optional)
    """
    model = YOLO(model_path)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Get image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(image_dir) 
                  if f.lower().endswith(image_extensions)]
    
    print(f"🔄 Running inference on {len(image_files)} images...")
    print("-" * 50)
    
    total_detections = 0
    
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        results = model(img_path, conf=conf, verbose=False)
        
        if results and len(results) > 0:
            result = results[0]
            num_detections = len(result.boxes)
            total_detections += num_detections
            
            print(f"✓ {img_file:<40} → {num_detections} detections")
            
            if output_dir:
                output_path = os.path.join(output_dir, f"det_{img_file}")
                result.save(output_path)
        else:
            print(f"○ {img_file:<40} → No detections")
    
    print("-" * 50)
    print(f"✅ Total detections: {total_detections}")
    
    if output_dir:
        print(f"📁 Results saved to {output_dir}")


def run_video_inference(model_path: str, video_path: str, conf: float = 0.5, output_path: str = None):
    """
    Run inference on video file.
    
    Args:
        model_path: Path to trained model
        video_path: Path to video file
        conf: Confidence threshold
        output_path: Path to save output video (optional)
    """
    model = YOLO(model_path)
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"🎬 Video: {video_path}")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Total frames: {total_frames}")
    print()
    
    # Setup video writer if output requested
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    total_detections = 0
    
    print("Processing frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run inference
        results = model(frame, conf=conf, verbose=False)
        
        # Annotate frame
        if results and len(results) > 0:
            result = results[0]
            frame = result.plot()
            total_detections += len(result.boxes)
        
        # Write frame
        if writer:
            writer.write(frame)
        
        # Print progress
        if frame_count % max(1, total_frames // 10) == 0:
            pct = 100 * frame_count / total_frames
            print(f"  {pct:.0f}% ({frame_count}/{total_frames})")
    
    cap.release()
    if writer:
        writer.release()
    
    print(f"\n✅ Processing complete!")
    print(f"   Frames processed: {frame_count}")
    print(f"   Total detections: {total_detections}")
    print(f"   Avg detections/frame: {total_detections/frame_count:.1f}")
    
    if output_path:
        print(f"   Output video: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run YOLOv11 inference")
    parser.add_argument("--model", required=True, help="Path to trained model (.pt)")
    parser.add_argument("--source", required=True, help="Path to image, directory, or video")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--output", help="Output directory for results")
    parser.add_argument("--save", action="store_true", help="Save output image")
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        return
    
    # Check source
    if not os.path.exists(args.source):
        print(f"❌ Source not found: {args.source}")
        return
    
    # Determine source type
    if os.path.isfile(args.source):
        ext = Path(args.source).suffix.lower()
        
        if ext in ('.jpg', '.jpeg', '.png', '.bmp', '.tiff'):
            # Single image
            run_image_inference(args.model, args.source, args.conf, args.save)
        
        elif ext in ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'):
            # Video
            run_video_inference(args.model, args.source, args.conf, args.output)
        
        else:
            print(f"❌ Unsupported file format: {ext}")
    
    elif os.path.isdir(args.source):
        # Directory of images
        run_batch_inference(args.model, args.source, args.conf, args.output)
    
    else:
        print(f"❌ Source is neither file nor directory: {args.source}")


if __name__ == "__main__":
    main()
