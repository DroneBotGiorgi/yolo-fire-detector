"""Export/import YOLO model artifacts via Drive filesystem or OAuth Google Drive API."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.file"]
DEFAULT_OAUTH_CREDENTIALS_FILE = "tools/model_registry/oauth_credentials.local.json"


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid YAML object in {path}")
    return payload


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_with_persistent_root(path_value: str, persistent_root: Path) -> Path:
    raw = Path(path_value)
    if raw.is_absolute():
        return raw
    return (persistent_root / raw).resolve()


def resolve_with_project_root(path_value: str) -> Path:
    raw = Path(path_value)
    if raw.is_absolute():
        return raw
    return (PROJECT_ROOT / raw).resolve()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid JSON object in {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def parse_client_config_from_bundle(bundle: dict[str, Any]) -> dict[str, Any] | None:
    client_secrets = bundle.get("client_secrets")
    if isinstance(client_secrets, dict):
        return client_secrets

    installed = bundle.get("installed")
    if isinstance(installed, dict):
        return {"installed": installed}

    return None


def parse_token_info_from_bundle(bundle: dict[str, Any]) -> dict[str, Any] | None:
    token_info = bundle.get("token")
    if isinstance(token_info, dict):
        return token_info

    # Backward-compatible support for token-only JSON files.
    if isinstance(bundle.get("access_token"), str) or isinstance(bundle.get("token"), str):
        return bundle

    return None


def slug_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "-", value.strip())
    token = re.sub(r"-{2,}", "-", token).strip("-").lower()
    return token or "item"


def build_import_suffix(registry_name: str, run_label: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    return f"imported-from-{slug_token(registry_name)}-{slug_token(run_label)}-{timestamp}"


def resolve_import_target_path(base_path: Path, overwrite: bool, suffix: str) -> Path:
    if overwrite or not base_path.exists():
        return base_path

    candidate = base_path.with_name(f"{base_path.stem}__{suffix}{base_path.suffix}")
    if not candidate.exists():
        return candidate

    index = 2
    while True:
        candidate = base_path.with_name(f"{base_path.stem}__{suffix}-{index}{base_path.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def build_drive_service(
    *,
    oauth_credentials_file: Path,
    client_secrets_path: Path | None,
    token_path: Path | None,
):
    """Build an authenticated Google Drive API service using OAuth."""
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError as ex:  # pragma: no cover - dependency guard
        raise ImportError(
            "Missing Google Drive OAuth dependencies. Install: "
            "google-api-python-client google-auth-httplib2 google-auth-oauthlib"
        ) from ex

    bundle: dict[str, Any] = {}
    if oauth_credentials_file.exists():
        bundle = read_json(oauth_credentials_file)

    creds = None
    token_info = parse_token_info_from_bundle(bundle)
    if token_info:
        creds = Credentials.from_authorized_user_info(token_info, DRIVE_SCOPES)
    elif token_path is not None and token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), DRIVE_SCOPES)

    client_config = parse_client_config_from_bundle(bundle)
    if client_config is None and client_secrets_path is not None and client_secrets_path.exists():
        client_config = read_json(client_secrets_path)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if client_config is None:
                raise FileNotFoundError(
                    "OAuth credentials non trovate. Usa --oauth-credentials-file con un JSON che contiene "
                    "'client_secrets' oppure passa --oauth-client-secrets (legacy)."
                )

            flow = InstalledAppFlow.from_client_config(client_config, DRIVE_SCOPES)
            creds = flow.run_local_server(port=0)

    merged_bundle = dict(bundle)
    if client_config is not None and "client_secrets" not in merged_bundle:
        merged_bundle["client_secrets"] = client_config
    merged_bundle["token"] = json.loads(creds.to_json())
    write_json(oauth_credentials_file, merged_bundle)

    if token_path is not None:
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(creds.to_json(), encoding="utf-8")

    return build("drive", "v3", credentials=creds)


def escape_drive_q(value: str) -> str:
    return value.replace("'", "\\'")


def find_drive_file_id(service, *, name: str, parent_id: str, mime_type: str | None = None) -> str | None:
    q_parts = [
        f"name = '{escape_drive_q(name)}'",
        f"'{parent_id}' in parents",
        "trashed = false",
    ]
    if mime_type:
        q_parts.append(f"mimeType = '{mime_type}'")

    response = service.files().list(
        q=" and ".join(q_parts),
        spaces="drive",
        fields="files(id, name, modifiedTime)",
        pageSize=10,
        orderBy="modifiedTime desc",
    ).execute()
    files = response.get("files", [])
    if not files:
        return None
    return str(files[0]["id"])


def ensure_drive_folder(service, *, name: str, parent_id: str) -> str:
    folder_mime = "application/vnd.google-apps.folder"
    existing = find_drive_file_id(service, name=name, parent_id=parent_id, mime_type=folder_mime)
    if existing:
        return existing

    metadata = {"name": name, "mimeType": folder_mime, "parents": [parent_id]}
    created = service.files().create(body=metadata, fields="id").execute()
    return str(created["id"])


def upload_drive_file(service, *, parent_id: str, source_path: Path, target_name: str) -> str:
    from googleapiclient.http import MediaFileUpload

    existing = find_drive_file_id(service, name=target_name, parent_id=parent_id)
    media = MediaFileUpload(str(source_path), resumable=True)
    if existing:
        updated = service.files().update(fileId=existing, media_body=media, fields="id").execute()
        return str(updated["id"])

    metadata = {"name": target_name, "parents": [parent_id]}
    created = service.files().create(body=metadata, media_body=media, fields="id").execute()
    return str(created["id"])


def upload_drive_text(service, *, parent_id: str, filename: str, text_payload: str) -> str:
    from googleapiclient.http import MediaIoBaseUpload

    existing = find_drive_file_id(service, name=filename, parent_id=parent_id)
    stream = io.BytesIO(text_payload.encode("utf-8"))
    media = MediaIoBaseUpload(stream, mimetype="text/yaml", resumable=False)

    if existing:
        updated = service.files().update(fileId=existing, media_body=media, fields="id").execute()
        return str(updated["id"])

    metadata = {"name": filename, "parents": [parent_id]}
    created = service.files().create(body=metadata, media_body=media, fields="id").execute()
    return str(created["id"])


def download_drive_file(service, *, file_id: str, target_path: Path) -> None:
    from googleapiclient.http import MediaIoBaseDownload

    target_path.parent.mkdir(parents=True, exist_ok=True)
    request = service.files().get_media(fileId=file_id)
    with target_path.open("wb") as handle:
        downloader = MediaIoBaseDownload(handle, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()


def read_drive_text(service, *, file_id: str) -> str:
    request = service.files().get_media(fileId=file_id)
    payload: bytes = request.execute()
    return payload.decode("utf-8")


def choose_local_artifacts(
    *,
    local_persistent_root: Path,
    model_path_arg: str | None,
    metadata_path_arg: str | None,
    run_label_arg: str | None,
) -> tuple[Path, Path | None, str]:
    exports_root = local_persistent_root / "exports"

    if model_path_arg:
        model_path = resolve_with_persistent_root(model_path_arg, local_persistent_root)
        metadata_path = (
            resolve_with_persistent_root(metadata_path_arg, local_persistent_root)
            if metadata_path_arg
            else None
        )
        run_label = run_label_arg or model_path.stem
        return model_path, metadata_path, run_label

    latest_path = exports_root / "latest.yaml"
    if not latest_path.exists():
        raise FileNotFoundError(
            f"Missing latest registry: {latest_path}. Pass --model-path explicitly or export a model first."
        )

    latest = read_yaml(latest_path)
    model_rel = latest.get("model_path")
    if not isinstance(model_rel, str) or not model_rel.strip():
        raise ValueError(f"latest.yaml does not contain a valid model_path: {latest_path}")

    metadata_rel = latest.get("metadata_path")
    run_label = run_label_arg or str(latest.get("run_label") or Path(model_rel).stem)

    model_path = resolve_with_persistent_root(model_rel, local_persistent_root)
    metadata_path = None
    if isinstance(metadata_rel, str) and metadata_rel.strip():
        metadata_candidate = resolve_with_persistent_root(metadata_rel, local_persistent_root)
        if metadata_candidate.exists():
            metadata_path = metadata_candidate

    return model_path, metadata_path, run_label


def export_to_drive(
    *,
    drive_root: Path,
    registry_name: str,
    local_persistent_root: Path,
    model_path_arg: str | None,
    metadata_path_arg: str | None,
    run_label_arg: str | None,
) -> None:
    model_path, metadata_path, run_label = choose_local_artifacts(
        local_persistent_root=local_persistent_root,
        model_path_arg=model_path_arg,
        metadata_path_arg=metadata_path_arg,
        run_label_arg=run_label_arg,
    )

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if metadata_path is not None and not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    registry_root = (drive_root / registry_name).resolve()
    version_dir = registry_root / "models" / run_label
    version_dir.mkdir(parents=True, exist_ok=True)

    target_model = version_dir / model_path.name
    shutil.copy2(model_path, target_model)

    target_metadata: Path | None = None
    if metadata_path is not None:
        target_metadata = version_dir / metadata_path.name
        shutil.copy2(metadata_path, target_metadata)

    model_hash = sha256_of_file(target_model)
    model_size = target_model.stat().st_size

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_label": run_label,
        "model_filename": target_model.name,
        "metadata_filename": target_metadata.name if target_metadata else None,
        "model_sha256": model_hash,
        "model_size_bytes": model_size,
    }
    write_yaml(version_dir / "model_manifest.yaml", manifest)

    latest = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "run_label": run_label,
        "manifest_path": f"models/{run_label}/model_manifest.yaml",
    }
    write_yaml(registry_root / "latest.yaml", latest)

    print(f"Export completed: {target_model}")
    print(f"SHA256: {model_hash}")
    print(f"Drive latest: {registry_root / 'latest.yaml'}")


def export_to_drive_oauth(
    *,
    service,
    drive_parent_id: str,
    registry_name: str,
    local_persistent_root: Path,
    model_path_arg: str | None,
    metadata_path_arg: str | None,
    run_label_arg: str | None,
) -> None:
    model_path, metadata_path, run_label = choose_local_artifacts(
        local_persistent_root=local_persistent_root,
        model_path_arg=model_path_arg,
        metadata_path_arg=metadata_path_arg,
        run_label_arg=run_label_arg,
    )

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if metadata_path is not None and not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    registry_id = ensure_drive_folder(service, name=registry_name, parent_id=drive_parent_id)
    models_id = ensure_drive_folder(service, name="models", parent_id=registry_id)
    run_dir_id = ensure_drive_folder(service, name=run_label, parent_id=models_id)

    upload_drive_file(service, parent_id=run_dir_id, source_path=model_path, target_name=model_path.name)
    if metadata_path is not None:
        upload_drive_file(service, parent_id=run_dir_id, source_path=metadata_path, target_name=metadata_path.name)

    model_hash = sha256_of_file(model_path)
    model_size = model_path.stat().st_size
    manifest_payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_label": run_label,
        "model_filename": model_path.name,
        "metadata_filename": metadata_path.name if metadata_path else None,
        "model_sha256": model_hash,
        "model_size_bytes": model_size,
    }
    upload_drive_text(
        service,
        parent_id=run_dir_id,
        filename="model_manifest.yaml",
        text_payload=yaml.safe_dump(manifest_payload, sort_keys=False, allow_unicode=False),
    )

    latest_payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "run_label": run_label,
        "manifest_path": f"models/{run_label}/model_manifest.yaml",
    }
    upload_drive_text(
        service,
        parent_id=registry_id,
        filename="latest.yaml",
        text_payload=yaml.safe_dump(latest_payload, sort_keys=False, allow_unicode=False),
    )

    print(f"Export completed via OAuth for run_label={run_label}")
    print(f"SHA256: {model_hash}")


def import_from_drive(
    *,
    drive_root: Path,
    registry_name: str,
    target_persistent_root: Path,
    run_label_arg: str | None,
    overwrite: bool,
) -> None:
    registry_root = (drive_root / registry_name).resolve()
    if not registry_root.exists():
        raise FileNotFoundError(f"Drive registry folder not found: {registry_root}")

    if run_label_arg:
        run_label = run_label_arg
    else:
        latest = read_yaml(registry_root / "latest.yaml")
        run_label = str(latest.get("run_label") or "").strip()
        if not run_label:
            raise ValueError("latest.yaml does not contain run_label")

    version_dir = registry_root / "models" / run_label
    manifest_path = version_dir / "model_manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing model manifest: {manifest_path}")

    manifest = read_yaml(manifest_path)
    model_filename = str(manifest.get("model_filename") or "").strip()
    metadata_filename = str(manifest.get("metadata_filename") or "").strip()
    expected_sha = str(manifest.get("model_sha256") or "").strip()
    if not model_filename:
        raise ValueError(f"Invalid model_filename in {manifest_path}")

    source_model = version_dir / model_filename
    if not source_model.exists():
        raise FileNotFoundError(f"Model file not found on drive: {source_model}")

    actual_sha = sha256_of_file(source_model)
    if expected_sha and actual_sha != expected_sha:
        raise ValueError(f"SHA256 mismatch for {source_model}. expected={expected_sha} actual={actual_sha}")

    exports_root = target_persistent_root / "exports"
    exports_root.mkdir(parents=True, exist_ok=True)

    import_suffix = build_import_suffix(registry_name, run_label)

    target_model = resolve_import_target_path(exports_root / model_filename, overwrite, import_suffix)
    shutil.copy2(source_model, target_model)

    target_metadata: Path | None = None
    if metadata_filename:
        source_metadata = version_dir / metadata_filename
        if source_metadata.exists():
            target_metadata = resolve_import_target_path(exports_root / metadata_filename, overwrite, import_suffix)
            shutil.copy2(source_metadata, target_metadata)

    latest_local = {
        "run_label": run_label,
        "model_path": f"exports/{target_model.name}",
        "metadata_path": f"exports/{target_metadata.name}" if target_metadata else None,
    }
    write_yaml(exports_root / "latest.yaml", latest_local)

    if target_model.name != model_filename:
        print(f"Model name conflict detected, saved as: {target_model.name}")
    if target_metadata is not None and target_metadata.name != metadata_filename:
        print(f"Metadata name conflict detected, saved as: {target_metadata.name}")

    print(f"Import completed: {target_model}")
    print(f"SHA256 verified: {actual_sha}")
    print(f"Local latest: {exports_root / 'latest.yaml'}")


def import_from_drive_oauth(
    *,
    service,
    drive_parent_id: str,
    registry_name: str,
    target_persistent_root: Path,
    run_label_arg: str | None,
    overwrite: bool,
) -> None:
    registry_id = find_drive_file_id(
        service,
        name=registry_name,
        parent_id=drive_parent_id,
        mime_type="application/vnd.google-apps.folder",
    )
    if not registry_id:
        raise FileNotFoundError(f"Drive registry folder not found: {registry_name}")

    if run_label_arg:
        run_label = run_label_arg
    else:
        latest_id = find_drive_file_id(service, name="latest.yaml", parent_id=registry_id)
        if not latest_id:
            raise FileNotFoundError("latest.yaml not found in drive registry")
        latest = yaml.safe_load(read_drive_text(service, file_id=latest_id)) or {}
        run_label = str(latest.get("run_label") or "").strip()
        if not run_label:
            raise ValueError("latest.yaml does not contain run_label")

    models_id = find_drive_file_id(
        service,
        name="models",
        parent_id=registry_id,
        mime_type="application/vnd.google-apps.folder",
    )
    if not models_id:
        raise FileNotFoundError("models folder not found in drive registry")

    run_dir_id = find_drive_file_id(
        service,
        name=run_label,
        parent_id=models_id,
        mime_type="application/vnd.google-apps.folder",
    )
    if not run_dir_id:
        raise FileNotFoundError(f"Run folder not found in drive registry: {run_label}")

    manifest_id = find_drive_file_id(service, name="model_manifest.yaml", parent_id=run_dir_id)
    if not manifest_id:
        raise FileNotFoundError(f"model_manifest.yaml not found for run: {run_label}")

    manifest = yaml.safe_load(read_drive_text(service, file_id=manifest_id)) or {}
    model_filename = str(manifest.get("model_filename") or "").strip()
    metadata_filename = str(manifest.get("metadata_filename") or "").strip()
    expected_sha = str(manifest.get("model_sha256") or "").strip()
    if not model_filename:
        raise ValueError("Invalid model_filename in model_manifest.yaml")

    model_id = find_drive_file_id(service, name=model_filename, parent_id=run_dir_id)
    if not model_id:
        raise FileNotFoundError(f"Model file not found on drive for run {run_label}: {model_filename}")

    exports_root = target_persistent_root / "exports"
    exports_root.mkdir(parents=True, exist_ok=True)
    import_suffix = build_import_suffix(registry_name, run_label)

    target_model = resolve_import_target_path(exports_root / model_filename, overwrite, import_suffix)
    download_drive_file(service, file_id=model_id, target_path=target_model)

    actual_sha = sha256_of_file(target_model)
    if expected_sha and actual_sha != expected_sha:
        target_model.unlink(missing_ok=True)
        raise ValueError(
            f"SHA256 mismatch for imported model. expected={expected_sha} actual={actual_sha}"
        )

    target_metadata: Path | None = None
    if metadata_filename:
        metadata_id = find_drive_file_id(service, name=metadata_filename, parent_id=run_dir_id)
        if metadata_id:
            target_metadata = resolve_import_target_path(exports_root / metadata_filename, overwrite, import_suffix)
            download_drive_file(service, file_id=metadata_id, target_path=target_metadata)

    latest_local = {
        "run_label": run_label,
        "model_path": f"exports/{target_model.name}",
        "metadata_path": f"exports/{target_metadata.name}" if target_metadata else None,
    }
    write_yaml(exports_root / "latest.yaml", latest_local)

    if target_model.name != model_filename:
        print(f"Model name conflict detected, saved as: {target_model.name}")
    if target_metadata is not None and target_metadata.name != metadata_filename:
        print(f"Metadata name conflict detected, saved as: {target_metadata.name}")

    print(f"Import completed: {target_model}")
    print(f"SHA256 verified: {actual_sha}")
    print(f"Local latest: {exports_root / 'latest.yaml'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync model artifacts with Google Drive (filesystem or OAuth)")
    parser.add_argument("command", choices=["export", "import"], help="Operation to execute")
    parser.add_argument(
        "--auth-mode",
        choices=["filesystem", "oauth"],
        default="filesystem",
        help="Drive access mode: local synced folder or OAuth API",
    )
    parser.add_argument(
        "--drive-root",
        type=str,
        default=os.environ.get("GOOGLE_DRIVE_ROOT", ""),
        help="Google Drive local root (filesystem mode only, ex: G:/My Drive).",
    )
    parser.add_argument(
        "--drive-parent-id",
        type=str,
        default=os.environ.get("GOOGLE_DRIVE_PARENT_ID", "root"),
        help="OAuth mode only: parent folder ID in Google Drive (default: root)",
    )
    parser.add_argument(
        "--oauth-credentials-file",
        type=str,
        default=DEFAULT_OAUTH_CREDENTIALS_FILE,
        help=(
            "OAuth mode: single local JSON file containing both 'client_secrets' "
            "and generated 'token'"
        ),
    )
    parser.add_argument(
        "--oauth-client-secrets",
        type=str,
        default="",
        help="OAuth mode legacy: path to Google client secrets JSON",
    )
    parser.add_argument(
        "--oauth-token-path",
        type=str,
        default="",
        help="OAuth mode legacy: optional token JSON path for backward compatibility",
    )
    parser.add_argument(
        "--registry-name",
        type=str,
        default="yolo-fire-detector-models",
        help="Subfolder name used as model registry inside drive root",
    )

    parser.add_argument(
        "--local-persistent-root",
        type=str,
        default="artifacts/local",
        help="Local persistent root used for export discovery",
    )
    parser.add_argument("--model-path", type=str, default="", help="Model path for export (optional)")
    parser.add_argument("--metadata-path", type=str, default="", help="Metadata path for export (optional)")
    parser.add_argument("--run-label", type=str, default="", help="Run label for export/import (optional)")

    parser.add_argument(
        "--target-persistent-root",
        type=str,
        default="artifacts/local",
        help="Target local persistent root used for import",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite local files on import")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_label = args.run_label.strip() or None
    model_path = args.model_path.strip() or None
    metadata_path = args.metadata_path.strip() or None

    if args.auth_mode == "filesystem":
        drive_root_raw = args.drive_root.strip()
        if not drive_root_raw:
            raise ValueError("Filesystem mode requires --drive-root (or GOOGLE_DRIVE_ROOT)")

        drive_root = Path(drive_root_raw).resolve()
        if not drive_root.exists():
            raise FileNotFoundError(f"Drive root not found: {drive_root}")

        if args.command == "export":
            export_to_drive(
                drive_root=drive_root,
                registry_name=args.registry_name,
                local_persistent_root=(PROJECT_ROOT / args.local_persistent_root).resolve(),
                model_path_arg=model_path,
                metadata_path_arg=metadata_path,
                run_label_arg=run_label,
            )
            return

        import_from_drive(
            drive_root=drive_root,
            registry_name=args.registry_name,
            target_persistent_root=(PROJECT_ROOT / args.target_persistent_root).resolve(),
            run_label_arg=run_label,
            overwrite=args.overwrite,
        )
        return

    client_secrets_arg = args.oauth_client_secrets.strip()
    token_path_arg = args.oauth_token_path.strip()

    service = build_drive_service(
        oauth_credentials_file=resolve_with_project_root(args.oauth_credentials_file),
        client_secrets_path=(resolve_with_project_root(client_secrets_arg) if client_secrets_arg else None),
        token_path=(resolve_with_project_root(token_path_arg) if token_path_arg else None),
    )
    drive_parent_id = args.drive_parent_id.strip() or "root"

    if args.command == "export":
        export_to_drive_oauth(
            service=service,
            drive_parent_id=drive_parent_id,
            registry_name=args.registry_name,
            local_persistent_root=(PROJECT_ROOT / args.local_persistent_root).resolve(),
            model_path_arg=model_path,
            metadata_path_arg=metadata_path,
            run_label_arg=run_label,
        )
        return

    import_from_drive_oauth(
        service=service,
        drive_parent_id=drive_parent_id,
        registry_name=args.registry_name,
        target_persistent_root=(PROJECT_ROOT / args.target_persistent_root).resolve(),
        run_label_arg=run_label,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
