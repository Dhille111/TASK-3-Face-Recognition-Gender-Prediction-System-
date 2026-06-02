from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import tensorflow as tf

from gender_predictor import summarize_gender

IMAGE_SIZE = (224, 224)
MODEL_PATH = Path("models/face_model.keras")
LABEL_MAP_PATH = Path("label_map.json")
HAAR_CASCADE_PATH = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"


def load_label_map(path: Path) -> Dict[int, str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing label map: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(key): value for key, value in raw.items()}


def load_model(path: Path) -> tf.keras.Model:
    if not path.exists():
        raise FileNotFoundError(f"Missing face model: {path}. Run train_model.py first.")
    return tf.keras.models.load_model(path)


def crop_with_margin(frame, x, y, w, h, margin_ratio: float = 0.2):
    height, width = frame.shape[:2]
    margin_x = int(w * margin_ratio)
    margin_y = int(h * margin_ratio)
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(width, x + w + margin_x)
    y2 = min(height, y + h + margin_y)
    return frame[y1:y2, x1:x2]


def preprocess_face(face_bgr) -> np.ndarray:
    resized = cv2.resize(face_bgr, IMAGE_SIZE)
    face_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    face_array = face_rgb.astype(np.float32) / 255.0
    return np.expand_dims(face_array, axis=0)


def annotate_frame(frame, x, y, w, h, label: str, confidence: float, gender_text: str) -> None:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    text_y = max(30, y - 15)
    cv2.putText(frame, f"{label} ({confidence * 100:.0f}%)", (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, gender_text, (x, text_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


def recognize(threshold: float = 0.55, camera_index: int = 0) -> None:
    model = load_model(MODEL_PATH)
    label_map = load_label_map(LABEL_MAP_PATH)
    cascade = cv2.CascadeClassifier(str(HAAR_CASCADE_PATH))
    if cascade.empty():
        raise RuntimeError(f"Could not load Haar cascade from {HAAR_CASCADE_PATH}")

    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError("Could not open the webcam")

    print("Press q to quit.")

    while True:
        ok, frame = capture.read()
        if not ok:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            face_crop = crop_with_margin(frame, x, y, w, h)
            if face_crop.size == 0:
                continue

            input_tensor = preprocess_face(face_crop)
            predictions = model.predict(input_tensor, verbose=0)[0]
            class_index = int(np.argmax(predictions))
            confidence = float(predictions[class_index])
            label = label_map.get(class_index, "Unknown") if confidence >= threshold else "Unknown"
            gender_text = summarize_gender(face_crop, label if label != "Unknown" else "")
            annotate_frame(frame, x, y, w, h, label, confidence, gender_text)
            break

        cv2.imshow("Face Recognition", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recognize registered faces from the webcam.")
    parser.add_argument("--threshold", type=float, default=0.55, help="Minimum confidence required to show a known name")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recognize(args.threshold, args.camera)


if __name__ == "__main__":
    main()
