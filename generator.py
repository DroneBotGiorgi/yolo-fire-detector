"""
YOLO Fire Detector Dataset Generator

Main entry point for synthetic dataset generation.

Architecture:
- settings.py: Configuration classes (ImageTransformSettings, DatasetGenerationSettings)
- transformations.py: Image augmentation and transformation functions
- utils.py: Utility functions (I/O, YOLO labels, folder management)
- generator.py: Main orchestration logic
"""

import cv2
import random

from settings import ImageTransformSettings, DatasetGenerationSettings
from transformations import (
    generate_random_background,
    augment_fire,
    resize_fire_with_alpha,
    add_shadow,
    alpha_composite,
    add_occlusion_from_background,
    augment_background,
)
from utils import (
    make_output_folders,
    load_fire_image,
    yolo_label_from_bbox,
    save_sample,
    show_demo,
)


# ============================================================
# ================== DATASET GENERATION ======================
# ============================================================


def generate_negative_sample(image_size: int) -> tuple:
    """
    Genera un'immagine negativa: solo sfondo casuale, nessun fuoco.
    In YOLO, per una negativa basta label vuota.
    
    Returns:
        Tuple[np.ndarray, str]: (immagine, label vuota)
    """
    bg = generate_random_background(image_size)

    if ImageTransformSettings.AUGMENT_NEGATIVE_BACKGROUNDS:
        bg = augment_background(bg)

    return bg, ""


def generate_positive_sample(fire_rgba, image_size: int) -> tuple:
    """
    Genera un'immagine positiva:
    - sfondo sintetico
    - fuoco augmentato
    - dimensione casuale
    - posizione casuale
    - ombra
    - occlusione
    
    Args:
        fire_rgba: Immagine del fuoco con alpha channel
        image_size: Dimensione dell'immagine di output
    
    Returns:
        Tuple[np.ndarray, str, Tuple[int, int, int, int]]: (immagine, label YOLO, bbox)
    """
    bg = generate_random_background(image_size)

    fire_aug = augment_fire(fire_rgba)

    # scala del fuoco
    scale = random.uniform(
        DatasetGenerationSettings.FIRE_SCALE_MIN,
        DatasetGenerationSettings.FIRE_SCALE_MAX
    )
    fire_aug = resize_fire_with_alpha(fire_aug, scale)

    fh, fw = fire_aug.shape[:2]

    # sicurezza: se il fuoco fosse più grande della canvas, riduco
    if fw >= image_size or fh >= image_size:
        safe_scale = min((image_size - 2) / max(1, fw), (image_size - 2) / max(1, fh))
        fire_aug = resize_fire_with_alpha(fire_aug, safe_scale)
        fh, fw = fire_aug.shape[:2]

    x = random.randint(0, image_size - fw)
    y = random.randint(0, image_size - fh)

    # ombra prima del compositing
    bg = add_shadow(bg, x, y, fw, fh)

    # compositing con alpha
    composed = alpha_composite(bg, fire_aug, x, y)

    # eventuale occlusione
    composed = add_occlusion_from_background(composed, x, y, fw, fh)

    # ulteriore rumore leggero sull'immagine finale (opzionale)
    if random.random() < 0.15:
        composed = augment_background(composed)

    label = yolo_label_from_bbox(x, y, fw, fh, image_size)
    bbox = (x, y, fw, fh)

    return composed, label, bbox


def main() -> None:
    """
    Orchestrazione principale della generazione del dataset.
    """
    dataset_root = DatasetGenerationSettings.DATASET_ROOT
    image_size = DatasetGenerationSettings.IMAGE_SIZE

    make_output_folders(dataset_root)

    fire = load_fire_image(DatasetGenerationSettings.FIRE_IMAGE_PATH)

    print("Avvio generazione dataset...")
    print(f"Immagine fuoco: {DatasetGenerationSettings.FIRE_IMAGE_PATH}")
    print(f"Cartella dataset: {dataset_root}")
    print(f"Numero immagini: {DatasetGenerationSettings.NUM_IMAGES}")
    print(f"Dimensione output: {image_size}x{image_size}")
    print(f"Negative ratio: {DatasetGenerationSettings.NEGATIVE_RATIO}")

    for i in range(DatasetGenerationSettings.NUM_IMAGES):

        is_negative = random.random() < DatasetGenerationSettings.NEGATIVE_RATIO

        if is_negative:
            image, label = generate_negative_sample(image_size)
            save_sample(
                image=image,
                label_text=label,
                dataset_root=dataset_root,
                index=i,
                train_split=DatasetGenerationSettings.TRAIN_SPLIT
            )

            if DatasetGenerationSettings.DEMO_MODE and i % 10 == 0:
                show_demo(image, bbox=None, wait_ms=DatasetGenerationSettings.DEMO_WAIT_MS)

        else:
            image, label, bbox = generate_positive_sample(fire, image_size)
            save_sample(
                image=image,
                label_text=label,
                dataset_root=dataset_root,
                index=i,
                train_split=DatasetGenerationSettings.TRAIN_SPLIT
            )

            if DatasetGenerationSettings.DEMO_MODE and i % 10 == 0:
                show_demo(image, bbox=bbox, wait_ms=DatasetGenerationSettings.DEMO_WAIT_MS)

        if (i + 1) % 100 == 0:
            print(f"Generate {i + 1}/{DatasetGenerationSettings.NUM_IMAGES} immagini")

    cv2.destroyAllWindows()
    print("Dataset generato correttamente.")


if __name__ == "__main__":
    main()