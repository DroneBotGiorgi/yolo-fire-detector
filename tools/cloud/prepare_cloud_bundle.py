"""Creates a zip bundle of the project for Colab or similar notebook services."""

import argparse
import json
from pathlib import Path
import zipfile


DEFAULT_EXCLUDES = {
    ".git",
    ".venv",
    ".vscode",
    ".ipynb_checkpoints",
    "__pycache__",
    "artifacts",
    "dataset",
    "detections",
    "fire_detector_runs",
    "runs",
    "dist",
    "env",
    "venv",
}

DEFAULT_INCLUDE_NAMES = {
    "README.md",
    "CLOUD_TRAINING.md",
    "TRAINING_PRESETS.md",
    "requirements.txt",
}

DEFAULT_INCLUDE_SUFFIXES = {
    ".py",
    ".ipynb",
    ".yaml",
    ".md",
}


def should_skip(path: Path, include_dataset: bool, include_runs: bool) -> bool:
    parts = set(path.parts)
    excludes = set(DEFAULT_EXCLUDES)
    if include_dataset:
        excludes.discard("dataset")
    if include_runs:
        excludes.discard("fire_detector_runs")
        excludes.discard("runs")
    return any(part in excludes for part in parts)


def should_include(relative_path: Path) -> bool:
    if relative_path.name in DEFAULT_INCLUDE_NAMES or relative_path.suffix.lower() in DEFAULT_INCLUDE_SUFFIXES:
        return True
    if relative_path.suffix.lower() == ".pt" and len(relative_path.parts) == 1:
        return True
    return "base_fire_images" in relative_path.parts and relative_path.suffix.lower() == ".png"


def strip_notebook_outputs(notebook_path: Path) -> bytes:
    """Return a UTF-8 notebook payload with code outputs removed."""
    payload = json.loads(notebook_path.read_text(encoding="utf-8-sig"))
    cells = payload.get("cells")
    if isinstance(cells, list):
        for cell in cells:
            if isinstance(cell, dict) and cell.get("cell_type") == "code":
                cell["outputs"] = []
                cell["execution_count"] = None
    return json.dumps(payload, indent=1, ensure_ascii=False).encode("utf-8")


def create_bundle(
    project_root: Path,
    output_path: Path,
    include_dataset: bool,
    include_runs: bool,
    strip_notebook_output: bool,
) -> None:
    files_added = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for candidate in project_root.rglob("*"):
            if candidate.is_dir():
                continue
            relative_path = candidate.relative_to(project_root)
            if should_skip(relative_path, include_dataset, include_runs):
                continue
            if not should_include(relative_path):
                continue
            if strip_notebook_output and candidate.suffix.lower() == ".ipynb":
                archive.writestr(str(relative_path), strip_notebook_outputs(candidate))
            else:
                archive.write(candidate, arcname=str(relative_path))
            files_added += 1

    bundle_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Bundle creato: {output_path}")
    print(f"File inclusi: {files_added}")
    print(f"Dimensione: {bundle_size_mb:.2f} MB")
    print("Passo successivo: carica lo zip in Colab o in Google Drive.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara uno zip del progetto per notebook cloud")
    parser.add_argument("--output", type=str, default="dist/yolo-fire-detector-cloud.zip")
    parser.add_argument("--include-dataset", action="store_true", help="Includi anche la cartella dataset se gia' generata")
    parser.add_argument("--include-runs", action="store_true", help="Includi checkpoint e risultati precedenti")
    parser.add_argument(
        "--keep-notebook-outputs",
        action="store_true",
        help="Mantieni gli output nei notebook inclusi nello zip (default: output rimossi)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    output_path = (project_root / args.output).resolve()
    create_bundle(
        project_root,
        output_path,
        args.include_dataset,
        args.include_runs,
        strip_notebook_output=not args.keep_notebook_outputs,
    )


if __name__ == "__main__":
    main()