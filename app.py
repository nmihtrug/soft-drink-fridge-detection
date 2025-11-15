import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
import io

@st.cache_resource
def load_model():
    return YOLO("weights/real/yolov13m.pt")

model = load_model()

st.title("🧱 Planogram 2D Generator (YOLOv13)")

uploaded_file = st.file_uploader("Upload ảnh kệ hàng (shelf image)", type=["jpg","jpeg","png"])

if uploaded_file:
    # read input
    img_pil = Image.open(uploaded_file).convert("RGB")
    img = np.array(img_pil)
    st.image(img, caption="Ảnh gốc", use_container_width=True)

    with st.spinner("Đang detect sản phẩm..."):
        results = model.predict(img, conf=0.40)[0]

    boxes = results.boxes
    names = model.names

    # ----------------------------
    # 1) Tách crop sản phẩm
    # ----------------------------
    items = []

    for b in boxes:
        cls = int(b.cls)
        label = names[cls]

        x1, y1, x2, y2 = map(int, b.xyxy[0])
        crop = img[y1:y2, x1:x2]

        items.append({
            "label": label,
            "crop": crop,
            "x_center": (x1 + x2)//2,
            "y_center": (y1 + y2)//2
        })

    # ----------------------------
    # 2) Sắp xếp thành cấu trúc hàng/cột
    # ----------------------------
    # Sắp xếp theo Y để xác định hàng
    items_sorted = sorted(items, key=lambda x: x["y_center"])

    # Tách thành các hàng bằng heuristic
    rows = []
    current_row = [items_sorted[0]]

    for i in range(1, len(items_sorted)):
        if abs(items_sorted[i]["y_center"] - items_sorted[i-1]["y_center"]) < 60:
            current_row.append(items_sorted[i])
        else:
            rows.append(current_row)
            current_row = [items_sorted[i]]
    rows.append(current_row)

    # Sắp xếp theo X trong từng hàng
    for r in rows:
        r.sort(key=lambda x: x["x_center"])

    # ----------------------------
    # 3) Tạo ảnh planogram 2D
    # ----------------------------
    cell_w = 200
    cell_h = 240
    margin = 20

    num_rows = len(rows)
    num_cols = max(len(r) for r in rows)

    planogram_w = num_cols * (cell_w + margin) + margin
    planogram_h = num_rows * (cell_h + margin) + margin

    planogram = np.ones((planogram_h, planogram_w, 3), dtype=np.uint8) * 255

    # Vẽ từng ô
    for r_idx, row in enumerate(rows):
        for c_idx, item in enumerate(row):
            # Resize crop
            crop = cv2.resize(item["crop"], (cell_w, cell_h-40))

            # Vị trí đặt ô
            x = margin + c_idx * (cell_w + margin)
            y = margin + r_idx * (cell_h + margin)

            # Đặt ảnh crop vào planogram
            planogram[y:y+cell_h-40, x:x+cell_w] = crop

            # Ghi tên sản phẩm
            cv2.putText(
                planogram,
                item["label"],
                (x + 5, y + cell_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

    st.subheader("📦 Planogram 2D được tạo:")
    st.image(planogram, use_container_width=True)

    # Download
    buf = io.BytesIO()
    Image.fromarray(planogram).save(buf, format="PNG")
    st.download_button("⬇️ Tải Planogram", buf.getvalue(), "planogram_2d.png", "image/png")
