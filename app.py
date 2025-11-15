import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
import json

# ============================
# Load YOLO
# ============================
@st.cache_resource
def load_model():
    return YOLO("weights/real/yolov11m.pt")

model = load_model()

# ============================
# Group shelves theo Y-center
# ============================
def group_shelves(items, threshold=80):
    shelves = []

    for item in items:
        placed = False
        for shelf in shelves:
            if abs(item["y_center"] - shelf[0]["y_center"]) < threshold:
                shelf.append(item)
                placed = True
                break
        if not placed:
            shelves.append([item])

    for shelf in shelves:
        shelf.sort(key=lambda x: x["x_center"])

    result = {}
    for i, shelf in enumerate(shelves):
        shelf_name = f"shelf_{i+1}"
        result[shelf_name] = []
        for col, item in enumerate(shelf, start=1):
            result[shelf_name].append({
                "col": col,
                "sku": item["label"],
                "crop": item.get("crop")
            })
    return result


# ============================
# Compare reference vs detected
# ============================
def compare_planogram(reference, detected):
    result = {}

    for shelf_name, ref_items in reference.items():
        det_items = detected.get(shelf_name, [])

        ref_map = {i["col"]: i["sku"] for i in ref_items}
        det_map = {i["col"]: i for i in det_items}

        max_col = max(
            max(ref_map.keys(), default=0),
            max(det_map.keys(), default=0)
        )

        shelf_result = {}
        for col in range(1, max_col + 1):
            ref_sku = ref_map.get(col)
            det_item = det_map.get(col)
            det_sku = det_item["sku"] if det_item else None

            if ref_sku and det_sku:
                status = "correct" if ref_sku == det_sku else "wrong"
            elif ref_sku and not det_sku:
                status = "missing"
            elif det_sku and not ref_sku:
                status = "extra"
            else:
                continue

            shelf_result[col] = {
                "ref": ref_sku,
                "det": det_sku,
                "crop": det_item["crop"] if det_item else None,
                "status": status
            }

        result[shelf_name] = shelf_result

    return result



# ============================
# RENDER PLANOGRAM REFERENCE (TEXT GRID)
# ============================
def render_reference_planogram(reference):
    cell_w, cell_h = 160, 80
    margin = 20

    shelves = list(reference.keys())
    max_col = max(len(reference[s]) for s in shelves)

    h = len(shelves) * (cell_h + margin) + margin
    w = max_col * (cell_w + margin) + margin

    img = np.ones((h, w, 3), dtype=np.uint8) * 255

    for row_idx, shelf_name in enumerate(shelves):
        for item in reference[shelf_name]:
            col = item["col"]
            sku = item["sku"]

            x1 = margin + (col - 1) * (cell_w + margin)
            y1 = margin + row_idx * (cell_h + margin)
            x2 = x1 + cell_w
            y2 = y1 + cell_h

            cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,0), 2)
            cv2.putText(img, sku, (x1 + 5, y1 + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    return img



# ============================
# RENDER DETECTED (REALISTIC) PLANOGRAM
# ============================
def render_detected_planogram(compare_result):
    cell_w, cell_h = 160, 160
    margin = 20

    shelves = list(compare_result.keys())
    max_col = max(len(compare_result[s]) for s in shelves)

    h = len(shelves) * (cell_h + margin) + margin
    w = max_col * (cell_w + margin) + margin

    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255

    colors = {
        "correct": (0, 200, 0),
        "wrong": (0, 0, 255),
        "missing": (180, 180, 180),
        "extra": (0, 140, 255)
    }

    for row_idx, shelf_name in enumerate(shelves):
        for col, info in compare_result[shelf_name].items():

            x1 = margin + (col - 1) * (cell_w + margin)
            y1 = margin + row_idx * (cell_h + margin)
            x2 = x1 + cell_w
            y2 = y1 + cell_h

            status = info["status"]
            color = colors[status]

            if status == "missing":
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)
                cv2.putText(canvas, f"exp:{info['ref']}", (x1+5, y1+80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                continue

            crop = info["crop"]
            if crop is not None:
                crop = cv2.resize(crop, (cell_w, cell_h))
                canvas[y1:y2, x1:x2] = crop

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)
            cv2.putText(canvas, info["det"], (x1+5, y1+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

            if status == "wrong":
                cv2.putText(canvas, f"exp:{info['ref']}", (x1+5, y1+45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)

    return canvas




# ============================
# STREAMLIT UI
# ============================
st.title("PLANOGRAM CHECKER")

uploaded_img = st.file_uploader("Chọn ảnh cần kiểm tra", type=["jpg","jpeg","png"])
reference_json = st.file_uploader("Tải file JSON planogram chuẩn", type=["json"])

if uploaded_img and reference_json:

    reference_data = json.load(reference_json)

    img_pil = Image.open(uploaded_img).convert("RGB")
    img = np.array(img_pil)

    # YOLO detect
    with st.spinner("Detecting..."):
        results = model.predict(img, conf=0.45)[0]

    # Extract detection + crop
    items = []
    for b in results.boxes:
        cls = int(b.cls)
        label = model.names[cls]
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        crop = img[y1:y2, x1:x2]

        items.append({
            "label": label,
            "x_center": (x1 + x2) // 2,
            "y_center": (y1 + y2) // 2,
            "crop": crop
        })

    detected_planogram = group_shelves(items)
    compare_result = compare_planogram(reference_data, detected_planogram)

    # Render 2 planograms
    ref_img = render_reference_planogram(reference_data)
    det_img = render_detected_planogram(compare_result)

    # ======= 2 CỘT SIDE BY SIDE =======
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Planogram Reference")
        st.image(ref_img, channels="RGB", use_container_width=True)

    with col2:
        st.subheader("Planogram thực tế")
        st.image(det_img, channels="RGB", use_container_width=True)

    # ======= SUMMARY / LEGEND =======
    st.markdown("## 📘 Định nghĩa kết quả")

    correct_count = 0
    wrong_count = 0
    missing_count = 0
    extra_count = 0

    for shelf in compare_result.values():
        for info in shelf.values():
            if info["status"] == "correct":
                correct_count += 1
            elif info["status"] == "wrong":
                wrong_count += 1
            elif info["status"] == "missing":
                missing_count += 1
            elif info["status"] == "extra":
                extra_count += 1

    st.markdown(f"""
    ### 👉 Tổng quan:
    - 🟩 **Đúng vị trí**: `{correct_count}`
    - 🟥 **Sai vị trí**: `{wrong_count}`
    - ⬜ **Thiếu sản phẩm**: `{missing_count}`
    - 🟧 **Thừa sản phẩm**: `{extra_count}`
    """)

    st.markdown("""
    ### 🎨 Giải thích màu sắc:
    - 🟩 **ĐÚNG** — SKU detect trùng hoàn toàn reference  
    - 🟥 **SAI** — SKU detect khác reference  
    - ⬜ **THIẾU** — Reference có nhưng detect không có  
    - 🟧 **THỪA** — Detect có nhưng reference không có  
    """)
