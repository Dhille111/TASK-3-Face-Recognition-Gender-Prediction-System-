from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2

try:
    import gender_guesser.detector as gender_detector
except Exception:  # pragma: no cover
    gender_detector = None

MODEL_DIR = Path("models")
GENDER_MODEL_PATH = MODEL_DIR / "gender_net.caffemodel"
GENDER_PROTO_PATH = MODEL_DIR / "gender_deploy.prototxt"
GENDER_LIST = ["Male", "Female"]

_CUSTOM_FEMALE_SUFFIXES = (
    "devi",
    "amma",
    "ammai",
    "akka",
    "priya",
    "latha",
    "lakshmi",
    "sri",
)
_CUSTOM_MALE_SUFFIXES = (
    "rao",
    "kumar",
    "babu",
    "singh",
)


def _normalize_name(name: str) -> str:
    return " ".join(name.replace("_", " ").replace("-", " ").split()).strip().lower()


def predict_gender_from_name(name: str) -> Tuple[Optional[str], float]:
    normalized = _normalize_name(name)
    if not normalized:
        return None, 0.0

    for suffix in _CUSTOM_FEMALE_SUFFIXES:
        if normalized.endswith(suffix):
            return "Female", 0.95

    for suffix in _CUSTOM_MALE_SUFFIXES:
        if normalized.endswith(suffix):
            return "Male", 0.95

    if gender_detector is None:
        return None, 0.0

    detector = gender_detector.Detector(case_sensitive=False)
    result = detector.get_gender(normalized.split()[0])

    if result in {"female", "mostly_female"}:
        return "Female", 0.80
    if result in {"male", "mostly_male"}:
        return "Male", 0.80

    return None, 0.0


def _load_gender_net() -> Optional[cv2.dnn_Net]:
    if not GENDER_MODEL_PATH.exists() or not GENDER_PROTO_PATH.exists():
        return None
    return cv2.dnn.readNet(str(GENDER_MODEL_PATH), str(GENDER_PROTO_PATH))


def predict_gender_from_face(face_bgr) -> Tuple[Optional[str], float]:
    net = _load_gender_net()
    if net is None or face_bgr is None:
        return None, 0.0

    blob = cv2.dnn.blobFromImage(
        face_bgr,
        scalefactor=1.0,
        size=(227, 227),
        mean=(78.4263377603, 87.7689143744, 114.895847746),
        swapRB=False,
        crop=False,
    )
    net.setInput(blob)
    preds = net.forward()
    class_index = int(preds[0].argmax())
    confidence = float(preds[0][class_index])
    return GENDER_LIST[class_index], confidence


def summarize_gender(face_bgr=None, name: str = "") -> str:
    face_gender, face_confidence = predict_gender_from_face(face_bgr)
    name_gender, name_confidence = predict_gender_from_name(name)

    parts = []
    if face_gender:
        parts.append(f"Face: {face_gender} {face_confidence * 100:.0f}%")
    if name_gender:
        parts.append(f"Name: {name_gender} {name_confidence * 100:.0f}%")

    if not parts:
        return "Gender: Unknown"
    return " | ".join(parts)
