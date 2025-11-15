"""
Evaluate ALL YOLOv11 models inside a folder (recursively)
with your original metric logic (TP/FP/FN matching)
and added:
- mAP50
- mAP50:95 (COCO-style IoU sweep)
"""

import os
import json
import argparse
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from ultralytics import YOLO


# ----------------------- IoU -----------------------
def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


# ----------------------- Load COCO JSON -----------------------
def load_coco_annotations(coco_json):
    with open(coco_json, "r") as f:
        data = json.load(f)
    images = {img["id"]: img for img in data["images"]}
    categories = {cat["id"]: cat["name"] for cat in data["categories"]}
    anns = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in anns:
            anns[img_id] = []
        x, y, w, h = ann["bbox"]
        anns[img_id].append({
            "bbox": [x, y, x + w, y + h],
            "category_id": ann["category_id"]
        })
    return images, categories, anns


# ----------------------- Confusion Matrix -----------------------
def plot_confusion_matrix(conf_matrix, class_names, output_path=None):
    plt.figure(figsize=(10, 8))
    plt.imshow(conf_matrix, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, conf_matrix[i][j], ha="center", va="center", color="black")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=120)
    plt.close()


# ----------------------- Compute AP -----------------------
def compute_ap(recalls, precisions):
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([1.0], precisions, [0.0]))

    # monotonic precision
    for i in range(len(precisions)-2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i+1])

    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    return ap


# ----------------------- Single model eval -----------------------
def evaluate_single_model(model_path, coco_json, images_dir,
                          iou_threshold, conf_threshold, output_dir):

    print(f"\n🚀 Evaluating: {model_path}")
    os.makedirs(output_dir, exist_ok=True)

    images, categories, gt_anns = load_coco_annotations(coco_json)
    num_classes = len(categories)
    class_order = sorted(categories.keys())  # mapping COCO ID → index
    class_names = [categories[cid] for cid in class_order]

    model = YOLO(model_path)

    # For mAP computing
    per_iou_scores = {thr: [[] for _ in range(num_classes)] for thr in np.arange(0.5, 1.0, 0.05)}
    per_iou_matches = {thr: [[] for _ in range(num_classes)] for thr in np.arange(0.5, 1.0, 0.05)}

    # For confusion matrix (at main iou_threshold)
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    TP = FP = FN = 0

    # Loop images
    for img_id, img_info in images.items():
        img_path = os.path.join(images_dir, img_info["file_name"])
        if not os.path.exists(img_path):
            continue

        gt_boxes = gt_anns.get(img_id, [])
        results = model(img_path, conf=conf_threshold, verbose=False)[0]

        preds = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            preds.append({"bbox": [x1, y1, x2, y2], "conf": conf, "cls": cls})

        # For each IoU threshold run matching
        for thr in np.arange(0.5, 1.0, 0.05):
            matched_gt = set()
            for pred in preds:
                best_iou = 0
                best_gt = None
                for i, gt in enumerate(gt_boxes):
                    iou_score = iou(pred["bbox"], gt["bbox"])
                    if iou_score > best_iou:
                        best_iou = iou_score
                        best_gt = (i, gt)

                pred_cls = pred["cls"]
                if best_iou >= thr:
                    gt_idx, gt = best_gt
                    gt_idx_global = class_order.index(gt["category_id"])
                    if gt_idx not in matched_gt:
                        per_iou_scores[thr][pred_cls].append(pred["conf"])
                        per_iou_matches[thr][pred_cls].append(1)
                        matched_gt.add(gt_idx)
                    else:
                        per_iou_scores[thr][pred_cls].append(pred["conf"])
                        per_iou_matches[thr][pred_cls].append(0)
                else:
                    per_iou_scores[thr][pred_cls].append(pred["conf"])
                    per_iou_matches[thr][pred_cls].append(0)

        # confusion matrix only at main iou_threshold
        matched_gt = set()
        for pred in preds:
            best_iou = 0
            best_gt = None
            for i, gt in enumerate(gt_boxes):
                iou_score = iou(pred["bbox"], gt["bbox"])
                if iou_score > best_iou:
                    best_iou = iou_score
                    best_gt = (i, gt)

            pred_cls = pred["cls"]
            if best_iou >= iou_threshold:
                gt_idx, gt = best_gt
                gt_cls_idx = class_order.index(gt["category_id"])
                if gt_idx not in matched_gt:
                    TP += 1
                    conf_matrix[gt_cls_idx][pred_cls] += 1
                    matched_gt.add(gt_idx)
                else:
                    FP += 1
            else:
                FP += 1
        FN += (len(gt_boxes) - len(matched_gt))

    # -------- Compute mAP50 & mAP50:95 --------
    aps_each_iou = []

    for thr in np.arange(0.5, 1.0, 0.05):
        aps = []
        for cls in range(num_classes):
            scores = np.array(per_iou_scores[thr][cls])
            matches = np.array(per_iou_matches[thr][cls])
            if len(scores) == 0:
                aps.append(0)
                continue

            order = np.argsort(-scores)
            scores = scores[order]
            matches = matches[order]

            tp_cum = np.cumsum(matches)
            fp_cum = np.cumsum(1 - matches)

            recalls = tp_cum / (tp_cum[-1] + (len(matches) - tp_cum[-1] + 1e-9))
            precisions = tp_cum / (tp_cum + fp_cum + 1e-9)
            aps.append(compute_ap(recalls, precisions))

        aps_each_iou.append(np.mean(aps))

    mAP50 = aps_each_iou[0]
    mAP50_95 = np.mean(aps_each_iou)

    precision = TP / (TP + FP + 1e-9)
    recall = TP / (TP + FN + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    # Save metrics
    metrics = {
        "mAP50": float(mAP50),
        "mAP50_95": float(mAP50_95),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "TP": TP, "FP": FP, "FN": FN
    }

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Save confusion matrix
    plot_confusion_matrix(conf_matrix, class_names,
                          os.path.join(output_dir, "confusion_matrix.png"))

    print(metrics)
    print(f"✓ Saved results to {output_dir}")


# ----------------------- Scan folder -----------------------
def evaluate_all(weights_dir, coco_json, images_dir, output_root, iou_threshold, conf_threshold):
    weights_dir = Path(weights_dir)
    all_models = list(weights_dir.rglob("*.pt"))

    print(f"\n🔍 Found {len(all_models)} models:")
    for m in all_models: print(" -", m)

    for model_path in all_models:
        rel = model_path.relative_to(weights_dir)
        out_name = str(rel).replace("/", "_").replace("\\", "_").replace(".pt", "")
        out_dir = Path(output_root) / out_name

        evaluate_single_model(
            str(model_path), coco_json, images_dir,
            iou_threshold, conf_threshold, str(out_dir)
        )


# ----------------------- CLI -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-dir", required=True)
    parser.add_argument("--coco-json", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output-dir", default="eval_all")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()

    evaluate_all(
        args.weights_dir, args.coco_json, args.images_dir,
        args.output_dir, args.iou, args.conf
    )


if __name__ == "__main__":
    main()
