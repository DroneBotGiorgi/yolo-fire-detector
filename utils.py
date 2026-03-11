"""
Utility functions for YOLO Fire Detector Dataset Generator

Contains helper functions for:
- File I/O operations (save samples)
- Folder structure creation
- YOLO label formatting
- Preview and demo
"""

import cv2
import numpy as np
import os
import random


def make_output_folders(dataset_root: str) -> None:
    """
    Crea la struttura classica YOLO:
    dataset/
        images/train
        images/val
        labels/train
        labels/val
    """
    folders = [
        os.path.join(dataset_root, "images", "train"),
        os.path.join(dataset_root, "images", "val"),
        os.path.join(dataset_root, "labels", "train"),
        os.path.join(dataset_root, "labels", "val"),
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)


def load_fire_image(path: str) -> np.ndarray:
    """
    Carica l'immagine PNG del fuoco.
    Se presente canale alpha, viene usato poi in compositing.
    """
    fire = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if fire is None:
        raise FileNotFoundError(
            f"Impossibile caricare l'immagine del fuoco dal path: {path}"
        )

    # Ammessi:
    # - PNG BGRA (4 canali)
    # - immagine BGR (3 canali)
    if len(fire.shape) != 3 or fire.shape[2] not in [3, 4]:
        raise ValueError(
            "L'immagine del fuoco deve avere 3 canali (BGR) o 4 canali (BGRA)."
        )

    return fire


def yolo_label_from_bbox(x: int, y: int, w: int, h: int, image_size: int, class_id: int = 0) -> str:
    """
    Converte bbox pixel -> formato YOLO:
    class_id center_x center_y width height
    con coordinate normalizzate in [0,1].
    """
    cx = (x + w / 2) / image_size
    cy = (y + h / 2) / image_size
    bw = w / image_size
    bh = h / image_size
    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def save_sample(image: np.ndarray, label_text: str, dataset_root: str, index: int, train_split: float) -> None:
    """
    Salva immagine e label nel sottoinsieme train oppure val.
    """
    split = "train" if random.random() < train_split else "val"

    img_path = os.path.join(dataset_root, "images", split, f"img_{index:05d}.jpg")
    txt_path = os.path.join(dataset_root, "labels", split, f"img_{index:05d}.txt")

    cv2.imwrite(img_path, image)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(label_text)


def show_demo(image: np.ndarray, bbox: tuple[int, int, int, int] | None = None, wait_ms: int = 150) -> None:
    """
    Preview veloce durante la generazione.
    Se bbox è presente, la disegna in verde.
    """
    preview = image.copy()

    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # riduco per sicurezza la visualizzazione
    preview_small = cv2.resize(preview, (700, 700))
    cv2.imshow("Generator preview", preview_small)
    cv2.waitKey(wait_ms)
