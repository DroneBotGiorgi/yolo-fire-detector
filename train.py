"""Training utilities for the YOLO Fire Detector project."""

import json
import os
import shutil

from ultralytics import YOLO

from settings import DatasetGenerationSettings, TrainingSettings


def create_dataset_yaml(dataset_root: str = DatasetGenerationSettings.DATASET_ROOT) -> str:
    """Crea il file data.yaml richiesto da YOLO."""
    yaml_path = os.path.join(dataset_root, "data.yaml")
    yaml_content = f"""path: {os.path.abspath(dataset_root)}
train: images/train
val: images/val

nc: 1
names: ['fire']
"""

    with open(yaml_path, "w", encoding="utf-8") as handle:
        handle.write(yaml_content)

    print(f"✓ Dataset YAML creato: {yaml_path}")
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
    export_dir: str | None,
    training_summary: dict,
) -> str:
    """Copia il modello finale e serializza i parametri usati."""
    resolved_export_dir = export_dir or os.path.join(run_dir, "final_export")
    os.makedirs(resolved_export_dir, exist_ok=True)

    best_weights_path = os.path.join(run_dir, "weights", "best.pt")
    final_model_path = os.path.join(resolved_export_dir, "best.pt")
    shutil.copy(best_weights_path, final_model_path)

    txt_path = os.path.join(resolved_export_dir, "training_settings.txt")
    with open(txt_path, "w", encoding="utf-8") as handle:
        handle.write("# Training settings utilizzate\n")
        for key, value in training_summary.items():
            handle.write(f"{key} = {value}\n")

    json_path = os.path.join(resolved_export_dir, "training_run.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(training_summary, handle, indent=2)

    print(f"📦 Export finale disponibile in: {resolved_export_dir}")
    return resolved_export_dir


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
    export_dir: str | None = None,
    extra_summary: dict | None = None,
) -> str:
    """Addestra il modello YOLO e restituisce la cartella di export finale."""
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

    summary = {
        "dataset_root": os.path.abspath(dataset_root),
        "project_name": project_name,
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

    final_export_dir = export_training_artifacts(run_dir, export_dir, summary)

    try:
        import google.colab  # type: ignore

        print("\n" + "=" * 60)
        print("GOOGLE COLAB - COMANDI DI DOWNLOAD")
        print("=" * 60)
        print("from google.colab import files")
        print(f"files.download('{final_export_dir}/best.pt')")
        print(f"files.download('{final_export_dir}/training_run.json')")
        print("=" * 60)
    except ImportError:
        pass

    return final_export_dir


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
