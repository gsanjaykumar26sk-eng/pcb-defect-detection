import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="PCB Defect Detection", layout="centered")

st.title("PCB Defect Detection (YOLOv8)")
st.write("Upload a PCB image to automatically detect and localize manufacturing defects.")
st.caption("Model trained on the DeepPCB dataset (binary trace-style images). Classes: open, short, mousebite, spur, copper, pin_hole.")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

confidence = st.slider("Confidence threshold", 0.1, 0.9, 0.25, 0.05)

uploaded_file = st.file_uploader("Upload a PCB image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Detecting defects..."):
        results = model.predict(np.array(image), conf=confidence)
        annotated = results[0].plot()[:, :, ::-1]  # BGR -> RGB

    st.image(annotated, caption="Detected defects", use_container_width=True)

    names = model.names
    counts = {}
    for c in results[0].boxes.cls:
        label = names[int(c)]
        counts[label] = counts.get(label, 0) + 1

    st.subheader("Summary")
    if counts:
        for k, v in counts.items():
            st.write(f"- **{k}**: {v}")
    else:
        st.write("No defects detected.")
