from __future__ import annotations

import argparse
from pathlib import Path

import cv2

DATA_DIR = Path("data/registered_faces")
HAAR_CASCADE_PATH = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"


def normalize_folder_name(name: str) -> str:
    return "_".join(name.strip().lower().split())


def crop_with_margin(frame, x, y, w, h, margin_ratio: float = 0.2):
    height, width = frame.shape[:2]
    margin_x = int(w * margin_ratio)
    margin_y = int(h * margin_ratio)
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(width, x + w + margin_x)
    y2 = min(height, y + h + margin_y)
    return frame[y1:y2, x1:x2]


def capture_faces(person_name: str, count: int = 100, camera_index: int = 0) -> None:
    folder_name = normalize_folder_name(person_name)
    output_dir = DATA_DIR / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    cascade = cv2.CascadeClassifier(str(HAAR_CASCADE_PATH))
    if cascade.empty():
        raise RuntimeError(f"Could not load Haar cascade from {HAAR_CASCADE_PATH}")

    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError("Could not open the webcam")

    captured = 0
    print("Press q to stop early.")

    while captured < count:
        ok, frame = capture.read()
        if not ok:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            face_crop = crop_with_margin(frame, x, y, w, h)
            if face_crop.size == 0:
                continue

            resized = cv2.resize(face_crop, (224, 224))
            file_path = output_dir / f"{folder_name}_{captured + 1:04d}.jpg"
            cv2.imwrite(str(file_path), resized)
            captured += 1

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{person_name} ({captured}/{count})",
                (x, max(30, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            break

        cv2.imshow("Capture Faces", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()
    print(f"Saved {captured} face images to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture registered face images from the webcam.")
    parser.add_argument("--name", required=True, help="Person name to store in data/registered_faces/<name>/")
    parser.add_argument("--count", type=int, default=100, help="Number of face images to capture")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    capture_faces(args.name, args.count, args.camera)


if __name__ == "__main__":
    main()
