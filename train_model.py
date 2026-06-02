from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import tensorflow as tf

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42
DATA_DIR = Path("data/registered_faces")
MODEL_PATH = Path("models/face_model.keras")
LABEL_MAP_PATH = Path("label_map.json")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def format_display_name(folder_name: str) -> str:
    return " ".join(folder_name.replace("_", " ").replace("-", " ").split()).title()


def find_image_files(data_dir: Path) -> Dict[str, List[Path]]:
    class_to_files: Dict[str, List[Path]] = defaultdict(list)
    for class_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        for file_path in sorted(class_dir.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                class_to_files[class_dir.name].append(file_path)
    return dict(class_to_files)


def split_items(items: Sequence[Path], seed: int = SEED) -> Tuple[List[Path], List[Path], List[Path]]:
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)

    if total <= 1:
        return shuffled, [], []
    if total == 2:
        return shuffled[:1], shuffled[1:], []

    train_count = max(1, int(total * 0.7))
    val_count = max(1, int(total * 0.15))
    test_count = total - train_count - val_count

    if test_count <= 0:
        test_count = 1
        if train_count > val_count:
            train_count -= 1
        else:
            val_count -= 1

    train_end = train_count
    val_end = train_end + val_count
    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def build_split_lists(class_to_files: Dict[str, List[Path]]) -> Tuple[List[Path], List[int], List[Path], List[int], List[Path], List[int], List[str]]:
    class_names = sorted(class_to_files)
    train_paths: List[Path] = []
    train_labels: List[int] = []
    val_paths: List[Path] = []
    val_labels: List[int] = []
    test_paths: List[Path] = []
    test_labels: List[int] = []

    for class_index, class_name in enumerate(class_names):
        train_items, val_items, test_items = split_items(class_to_files[class_name])
        train_paths.extend(train_items)
        train_labels.extend([class_index] * len(train_items))
        val_paths.extend(val_items)
        val_labels.extend([class_index] * len(val_items))
        test_paths.extend(test_items)
        test_labels.extend([class_index] * len(test_items))

    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, class_names


def load_image(path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def make_dataset(paths: Sequence[Path], labels: Sequence[int], training: bool) -> tf.data.Dataset:
    if not paths:
        return tf.data.Dataset.from_tensor_slices(([], [])).take(0)

    path_strings = [str(path) for path in paths]
    label_array = np.asarray(labels, dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((path_strings, label_array))
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        dataset = dataset.shuffle(buffer_size=max(8, len(path_strings)), seed=SEED, reshuffle_each_iteration=True)
    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def build_model(num_classes: int) -> tf.keras.Model:
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.12),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )

    base_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3),
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x * 255.0)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="face_recognition_model")


def save_label_map(class_names: Sequence[str]) -> None:
    label_map = {str(index): format_display_name(class_name) for index, class_name in enumerate(class_names)}
    LABEL_MAP_PATH.write_text(json.dumps(label_map, indent=2), encoding="utf-8")


def train(data_dir: Path) -> None:
    class_to_files = find_image_files(data_dir)
    if not class_to_files:
        raise RuntimeError(f"No training images found under {data_dir}")

    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, class_names = build_split_lists(class_to_files)
    if not train_paths:
        raise RuntimeError("Training split is empty. Add more face images before training.")

    train_ds = make_dataset(train_paths, train_labels, training=True)
    val_ds = make_dataset(val_paths, val_labels, training=False) if val_paths else None
    test_ds = make_dataset(test_paths, test_labels, training=False) if test_paths else None

    model = build_model(num_classes=len(class_names))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy" if val_ds is not None else "loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy" if val_ds is not None else "loss", factor=0.5, patience=2),
        tf.keras.callbacks.ModelCheckpoint(str(MODEL_PATH), monitor="val_accuracy" if val_ds is not None else "loss", save_best_only=True),
    ]

    fit_kwargs = {
        "epochs": 20,
        "callbacks": callbacks,
    }
    if val_ds is not None:
        fit_kwargs["validation_data"] = val_ds

    history = model.fit(train_ds, **fit_kwargs)

    if test_ds is not None:
        loss, accuracy = model.evaluate(test_ds, verbose=0)
        print(f"Test loss: {loss:.4f}")
        print(f"Test accuracy: {accuracy:.4f}")
    else:
        print("Test split is empty, so no held-out evaluation was run.")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    save_label_map(class_names)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved label map to {LABEL_MAP_PATH}")
    print(f"Classes: {', '.join(format_display_name(name) for name in class_names)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the face recognition model.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Root folder containing registered face folders")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(args.data_dir)


if __name__ == "__main__":
    main()
