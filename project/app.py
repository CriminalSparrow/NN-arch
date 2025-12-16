import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import DetrForObjectDetection, DetrImageProcessor


@st.cache_resource
def load_model(model_path, processor_path, device='cuda'):
    processor = DetrImageProcessor.from_pretrained(processor_path)
    model = DetrForObjectDetection.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return model, processor


def predict_and_visualize(image_pil, model, processor, class2id, device='cuda', threshold=0.5):
    # конвертируем PIL в tensor
    encoding = processor(images=image_pil, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**encoding)

    # получаем размеры исходного изображения
    w, h = image_pil.size
    target_sizes = torch.tensor([[h, w]]).to(device)

    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=threshold
    )[0]

    boxes = results["boxes"].cpu().numpy()
    labels = results["labels"].cpu().numpy()
    scores = results["scores"].cpu().numpy()

    id2label = {v: k for k, v in class2id.items()}
    img_vis = np.array(image_pil).copy()

    for box, cls, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img_vis,
            f"{id2label[cls]} {score:.2f}",
            (x1, max(0, y1-5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

    return Image.fromarray(img_vis)


device = "cuda" if torch.cuda.is_available() else "cpu"
model, processor = load_model(
    model_path="/app/detr-best-epoch1",
    processor_path="/app/detr-best-epoch1",
    device=device
)

class2id = {
    "vehicle": 0,
    "bus": 1,
    "bicycle": 2,
    'pedestrian': 3,
    'engine': 4,
    'truck': 5,
    'tricycle': 6,
    'obstacle': 7,
}

st.title("DETR Инференс: перекрестки")

uploaded_file = st.file_uploader("Загрузите изображение перекрестка", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Входное изображение", use_column_width=True)

    st.write("Обрабатываем моделью DETR...")
    result_img = predict_and_visualize(image, model, processor, class2id, device)

    st.image(result_img, caption="Предсказания DETR", use_column_width=True)
