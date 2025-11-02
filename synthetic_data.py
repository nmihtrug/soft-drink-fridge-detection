import json
import os
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from copy import deepcopy
from PIL import Image, ImageFilter


def _round_box(b, W, H):
    """Convert COCO bbox [x, y, w, h] to integer XYXY, clamp to image bounds."""
    x, y, w, h = b
    x0 = max(0, int(round(x)))
    y0 = max(0, int(round(y)))
    x1 = min(W, int(round(x + w)))
    y1 = min(H, int(round(y + h)))
    if x1 <= x0 or y1 <= y0:
        return None  # invalid after rounding/clamping
    return (x0, y0, x1, y1)


def _paste_with_optional_feather(canvas, patch, box_xyxy, feather=0):
    """Paste patch into canvas at box_xyxy with optional Gaussian blur feathering."""
    x0, y0, x1, y1 = box_xyxy
    patch = patch.resize((x1 - x0, y1 - y0), Image.BILINEAR)

    if feather <= 0:
        canvas.paste(patch, (x0, y0))
        return

    # Soft edges to hide seams
    mask = Image.new("L", (x1 - x0, y1 - y0), 255)
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))
    canvas.paste(patch, (x0, y0), mask)


def swap_coco_crops(
    img: Image.Image,
    annotations: List[Dict],
    *,
    use_boxes: Optional[List[int]] = None,
    seed: Optional[int] = None,
    resize_mode: str = "stretch",   # "stretch" or "fit"
    feather: int = 0,               # pixels of Gaussian blur on edges
    swap_labels: bool = True        # also swap class labels (category_id / category_name)
) -> Tuple[Image.Image, List[Dict]]:
    """
    Swap bounding box crops between objects in a COCO-annotated image.
    
    Args:
        img: PIL.Image for a single image
        annotations: list of COCO-style annos for this image (dicts with 'bbox', 'iscrowd', etc.)
        use_boxes: optional list of indices into annotations to include; default = all iscrowd==0
        seed: random seed for reproducible swaps
        resize_mode:
            - "stretch": source crop is resized to exactly fill the destination bbox
            - "fit": keep aspect ratio, letterbox into destination bbox
        feather: edge blur (0 = hard paste)
        swap_labels: if True, category labels follow swapped crops
    
    Returns:
        Tuple of (augmented PIL.Image, updated annotations list)
    """
    if seed is not None:
        random.seed(seed)

    W, H = img.size
    annos = annotations if use_boxes is None else [annotations[i] for i in use_boxes]
    cand_idx = [i for i, a in enumerate(annos) if (a.get("iscrowd", 0) == 0 and a.get("bbox"))]

    # Need at least 2 boxes to swap
    if len(cand_idx) < 2:
        return img.copy(), deepcopy(annotations)

    # Compute valid integer XYXY boxes and skip degenerate ones
    boxes_xyxy = []
    kept_idx = []
    for i in cand_idx:
        xyxy = _round_box(annos[i]["bbox"], W, H)
        if xyxy:
            boxes_xyxy.append(xyxy)
            kept_idx.append(i)

    if len(kept_idx) < 2:
        return img.copy(), deepcopy(annotations)

    # Extract crops from original image for all kept boxes
    crops = []
    for xyxy in boxes_xyxy:
        x0, y0, x1, y1 = xyxy
        crops.append(img.crop((x0, y0, x1, y1)))

    # Build a non-identity random permutation over kept boxes
    perm = list(range(len(crops)))
    while True:
        random.shuffle(perm)
        if any(perm[i] != i for i in range(len(perm))):
            break

    # Start with a copy of the original image
    out = img.copy()

    # Paste crops to destinations according to permutation
    for dst_i, src_i in enumerate(perm):
        dst_box = boxes_xyxy[dst_i]
        src_crop = crops[src_i]

        if resize_mode == "stretch":
            _paste_with_optional_feather(out, src_crop, dst_box, feather=feather)
        else:  # "fit": keep aspect, letterbox
            x0, y0, x1, y1 = dst_box
            dw, dh = x1 - x0, y1 - y0
            sw, sh = src_crop.size
            if sw == 0 or sh == 0 or dw <= 0 or dh <= 0:
                continue
            scale = min(dw / sw, dh / sh)
            nw, nh = max(1, int(sw * scale)), max(1, int(sh * scale))
            resized = src_crop.resize((nw, nh), Image.BILINEAR)
            # center in destination
            ox = x0 + (dw - nw) // 2
            oy = y0 + (dh - nh) // 2
            if feather > 0:
                mask = Image.new("L", (nw, nh), 255).filter(ImageFilter.GaussianBlur(radius=feather))
                out.paste(resized, (ox, oy), mask)
            else:
                out.paste(resized, (ox, oy))

    # Prepare updated annotations
    new_annotations = deepcopy(annotations)

    if swap_labels:
        # Gather labels aligned with kept boxes (source labels)
        source_labels_id = []
        source_labels_name = []
        for i in kept_idx:
            a = annos[i]
            source_labels_id.append(a.get("category_id"))
            source_labels_name.append(a.get("category_name"))  # optional

        # Helper: translate index-in-annos -> index-in-original-annotations
        def to_orig_index(k_in_annos: int) -> int:
            if use_boxes is None:
                return k_in_annos
            return use_boxes[k_in_annos]

        # Reassign labels: destination kept box receives label from source box per permutation
        for dst_i, src_i in enumerate(perm):
            orig_dst = to_orig_index(kept_idx[dst_i])
            # category_id always updated if present
            if source_labels_id[src_i] is not None:
                new_annotations[orig_dst]["category_id"] = source_labels_id[src_i]
            # also carry optional category_name if user stored it
            if source_labels_name[src_i] is not None:
                new_annotations[orig_dst]["category_name"] = source_labels_name[src_i]

    return out, new_annotations


def generate_synthetic_dataset(
    input_root: str,
    output_root: str,
    data_types: List[str] = None,
    num_aug_per_image: int = 20,
    resize_mode: str = "stretch",
    feather: int = 1
):
    """
    Generate synthetic augmented dataset using crop swapping.
    
    Args:
        input_root: Path to original dataset with COCO annotations
        output_root: Path to save synthetic dataset
        data_types: List of data splits to process (train, valid, test)
        num_aug_per_image: Number of synthetic images per original
        resize_mode: "stretch" or "fit"
        feather: Gaussian blur radius for edge feathering
    """
    if data_types is None:
        data_types = ['train', 'valid', 'test']
    
    for data_type in data_types:
        ann_file = os.path.join(input_root, data_type, '_annotations.coco.json')
        
        if not os.path.exists(ann_file):
            print(f"⚠ Skipping {data_type}: {ann_file} not found")
            continue
        
        with open(ann_file, 'r') as file:
            data = json.load(file)

        print(f"Processing {data_type} ({len(data['images'])} images)")

        # Prepare new lists for synthetic data
        new_images = []
        new_annotations = []
        next_img_id = max(img['id'] for img in data['images']) + 1
        next_ann_id = max(ann['id'] for ann in data['annotations']) + 1 if data['annotations'] else 1

        for image in data['images']:
            img_path = os.path.join(input_root, data_type, image["file_name"])
            
            if not os.path.exists(img_path):
                print(f"  ⚠ Image not found: {img_path}")
                continue
            
            img = Image.open(img_path).convert("RGB")
            anns = [anno for anno in data['annotations'] if anno['image_id'] == image['id']]

            for k in range(num_aug_per_image):
                aug_img, aug_anns = swap_coco_crops(
                    img, anns,
                    resize_mode=resize_mode,
                    feather=feather,
                    swap_labels=True
                )

                # Generate synthetic filename and save
                base_name = os.path.splitext(image['file_name'])[0]
                aug_filename = f"{base_name}_swap{k}.jpg"
                aug_path = os.path.join(output_root, data_type, aug_filename)
                os.makedirs(os.path.dirname(aug_path), exist_ok=True)
                aug_img.save(aug_path, quality=95)

                # Add to COCO metadata
                new_image_entry = {
                    "id": next_img_id,
                    "file_name": aug_filename,
                    "width": image["width"],
                    "height": image["height"]
                }
                new_images.append(new_image_entry)

                for ann in aug_anns:
                    ann_copy = dict(ann)
                    ann_copy["id"] = next_ann_id
                    ann_copy["image_id"] = next_img_id
                    new_annotations.append(ann_copy)
                    next_ann_id += 1

                next_img_id += 1

        # Merge original + synthetic data
        synthetic_coco = {
            "images": data['images'] + new_images,
            "annotations": data['annotations'] + new_annotations,
            "categories": data.get('categories', []),
            "licenses": data.get('licenses', []),
            "info": data.get('info', {})
        }

        # Save updated COCO JSON
        out_json = os.path.join(output_root, data_type, "_annotations.coco.json")
        with open(out_json, 'w') as f:
            json.dump(synthetic_coco, f, indent=2)

        print(f"✓ Saved {len(new_images)} synthetic images to {out_json}")
        print(f"  Total: {len(synthetic_coco['images'])} images, {len(synthetic_coco['annotations'])} annotations\n")


if __name__ == "__main__":
    # Example usage
    generate_synthetic_dataset(
        input_root="data/original",
        output_root="data/synthetic",
        num_aug_per_image=20
    )
