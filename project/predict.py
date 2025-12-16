"""Инференс модели DETR для детекции объектов на изображениях перекрестков
из консоли"""

import argparse
import os
import torch
import cv2
from PIL import Image
from transformers import DetrForObjectDetection, DetrImageProcessor

import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*meta parameter.*"
)


def parse_args():
    """Парсер аргументов командной строки"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str,
                        default="""test_images/10110010.jpg""",
                        help="Path to input image")
    parser.add_argument("--model", type=str,
                        default="detr-best-epoch1", help="Path to model dir")
    parser.add_argument("--out_dir", type=str,
                        default="predictions", help="Output directory")
    parser.add_argument("--threshold", type=float,
                        default=0.5, help="Score threshold")
    return parser.parse_args()


CLASS2ID = {
    "vehicle": 0,
    "bus": 1,
    "bicycle": 2,
    "pedestrian": 3,
    "engine": 4,
    "truck": 5,
    "tricycle": 6,
    "obstacle": 7,
}
ID2LABEL = {v: k for k, v in CLASS2ID.items()}


def main():
    """
    Загружаем модель, открываем изображение,
    делаем предсказание и сохраняем результат
    """
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # проверка путей
    assert os.path.exists(args.image), f"Image not found: {args.image}"
    assert os.path.exists(args.model), f"Model dir not found: {args.model}"

    os.makedirs(args.out_dir, exist_ok=True)

    # загрузка модели
    processor = DetrImageProcessor.from_pretrained(args.model)
    model = DetrForObjectDetection.from_pretrained(args.model)
    model.to(device)
    model.eval()

    # загрузка изображения
    image_pil = Image.open(args.image).convert("RGB")
    image_cv = cv2.imread(args.image)
    orig_h, orig_w = image_cv.shape[:2]

    # предпроцессинг
    encoding = processor(images=image_pil, return_tensors="pt")
    encoding = {k: v.to(device) for k, v in encoding.items()}

    # предсказание
    with torch.no_grad():
        outputs = model(**encoding)

    # пост процессинг
    target_sizes = torch.tensor([[orig_h, orig_w]]).to(device)

    results = processor.post_process_object_detection(
        outputs,
        target_sizes=target_sizes,
        threshold=args.threshold
    )[0]

    boxes = results["boxes"].cpu().numpy()
    labels = results["labels"].cpu().numpy()
    scores = results["scores"].cpu().numpy()

    # визуализация
    for box, cls, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.astype(int)

        cv2.rectangle(
            image_cv,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            image_cv,
            f"{ID2LABEL.get(cls, cls)} {score:.2f}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

    # сохранение
    out_path = os.path.join(
        args.out_dir,
        os.path.basename(args.image)
    )

    cv2.imwrite(out_path, image_cv)

    print(f"[OK] Saved prediction to: {out_path}")


if __name__ == "__main__":
    main()
