"""Training utilities for the YOLO Fire Detector project."""

import os
from pathlib import Path

import yaml

from ultralytics import YOLO

from settings import DatasetGenerationSettings, TrainingSettings


def portable_path(path: str | Path, root: str | Path | None = None) -> str:
    """Serialize a path relative to a chosen root when possible."""
    resolved_path = Path(path).resolve()
    if root is not None:
        try:
            return resolved_path.relative_to(Path(root).resolve()).as_posix()
        except ValueError:
            pass
    return resolved_path.as_posix()


def create_dataset_yaml(dataset_root: str = DatasetGenerationSettings.DATASET_ROOT) -> str:
    """Crea il file yolo_dataset.yaml richiesto da YOLO."""
    yaml_path = os.path.join(dataset_root, "yolo_dataset.yaml")
    dataset_path = Path(os.path.relpath(os.path.abspath(dataset_root), os.getcwd())).as_posix()
    with open(yaml_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "path": dataset_path,
                "train": "images/train",
                "val": "images/val",
                "nc": 1,
                "names": ["fire"],
            },
            handle,
            sort_keys=False,
            allow_unicode=False,
        )

    print(f"✓ Config dataset YOLO creata: {yaml_path}")
    return yaml_path


def validate_dataset(dataset_root: str) -> None:
    """Verifica che il dataset esista e contenga immagini di training."""
    if not os.path.exists(dataset_root):
        raise FileNotFoundError(
            f"Dataset non trovato in {dataset_root}\n"
            "Esegui prima: python run_experiment.py --config configs/local.default.yaml"
        )

    images_train = os.path.join(dataset_root, "images", "train")
    if not os.path.exists(images_train) or not os.listdir(images_train):
        raise FileNotFoundError(
            f"Nessuna immagine di training trovata in {images_train}\n"
            "Esegui prima: python run_experiment.py --config configs/local.default.yaml"
        )


def export_training_artifacts(
    run_dir: str,
    training_summary: dict,
) -> str:
    """Serializza i parametri effettivi del training nella root della run."""
    yaml_path = os.path.join(run_dir, "training_run.yaml")
    with open(yaml_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(training_summary, handle, sort_keys=False, allow_unicode=False)

    print(f"🧾 Metadata training scritti in: {yaml_path}")
    return yaml_path


def train_model(
    model_size: str = TrainingSettings.MODEL_SIZE,
    epochs: int = TrainingSettings.EPOCHS,
    batch_size: int = TrainingSettings.BATCH_SIZE,
    image_size: int = TrainingSettings.IMAGE_SIZE,
    device: str = TrainingSettings.DEVICE,
    resume: bool = False,
    dataset_root: str = DatasetGenerationSettings.DATASET_ROOT,
    project_name: str = TrainingSettings.PROJECT_NAME,
    experiment_name: str = TrainingSettings.EXPERIMENT_NAME,
    weights: str | None = None,
    extra_summary: dict | None = None,
) -> str:
    """Addestra il modello YOLO e restituisce la cartella della run."""
    validate_dataset(dataset_root)
    yaml_path = create_dataset_yaml(dataset_root)

    run_dir = os.path.join(project_name, experiment_name)
    checkpoint_path = os.path.join(run_dir, "weights", "last.pt")
    base_weights = weights or f"yolov8{model_size}.pt"

    print("\n" + "=" * 60)
    print("Preparazione training...")
    print("=" * 60)
    print(f"Dataset: {dataset_root}")
    print(f"Weights iniziali: {base_weights}")
    print(f"Output run: {run_dir}")

    if resume and os.path.exists(checkpoint_path):
        print(f"🔁 Resume da checkpoint: {checkpoint_path}")
        model = YOLO(checkpoint_path)
        model.train(resume=True)
    else:
        if resume:
            print("⚠️ Resume richiesto ma checkpoint non trovato, avvio da zero.")

        model = YOLO(base_weights)
        model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=image_size,
            batch=batch_size,
            patience=TrainingSettings.PATIENCE,
            device=device,
            project=project_name,
            name=experiment_name,
            exist_ok=TrainingSettings.OVERWRITE_EXISTING,
            verbose=TrainingSettings.VERBOSE,
            lr0=TrainingSettings.LEARNING_RATE_INIT,
            lrf=TrainingSettings.LEARNING_RATE_FINAL,
            momentum=TrainingSettings.MOMENTUM,
            weight_decay=TrainingSettings.WEIGHT_DECAY,
            degrees=TrainingSettings.ROTATION_DEGREES,
            translate=TrainingSettings.TRANSLATE,
            scale=TrainingSettings.SCALE,
            flipud=TrainingSettings.FLIP_VERTICAL,
            fliplr=TrainingSettings.FLIP_HORIZONTAL,
            mosaic=TrainingSettings.MOSAIC,
            amp=TrainingSettings.MIXED_PRECISION,
        )

    print("\n" + "=" * 60)
    print("Training completato")
    print("=" * 60)

    persistent_root = Path(project_name).resolve().parent
    summary = {
        "dataset_root": portable_path(dataset_root, persistent_root),
        "project_name": portable_path(project_name, persistent_root),
        "experiment_name": experiment_name,
        "model_size": model_size,
        "weights": base_weights,
        "epochs": epochs,
        "batch_size": batch_size,
        "image_size": image_size,
        "device": device,
        "resume": resume,
    }
    if extra_summary:
        summary.update(extra_summary)
    for attr in dir(TrainingSettings):
        if attr.isupper():
            summary[f"TrainingSettings.{attr}"] = getattr(TrainingSettings, attr)

    export_training_artifacts(run_dir, summary)
    return run_dir


def validate_model(
    model_path: str = "fire_detector_runs/train/weights/best.pt",
    dataset_root: str = DatasetGenerationSettings.DATASET_ROOT,
) -> None:
    """Valida un modello gia' addestrato sul dataset corrente."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modello non trovato: {model_path}")

    validate_dataset(dataset_root)
    yaml_path = create_dataset_yaml(dataset_root)

    print("\n" + "=" * 60)
    print("Validazione del modello...")
    print("=" * 60)

    model = YOLO(model_path)
    metrics = model.val(data=yaml_path)

    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
