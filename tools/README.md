# Tools

Utility non-core del progetto, tenute fuori dalla root per non confondere gli entrypoint principali.

## Cloud

- `tools/cloud/prepare_cloud_bundle.py`: crea lo zip da caricare in Colab o Drive

## Dataset

- `tools/dataset/dataset_viewer.py`: visualizza una griglia di sample del dataset con label YOLO

## Streaming

- `tools/streaming/fake_rtmp_webcam.py`: pubblica una webcam locale come stream RTMP/HLS per testare `detect.py`
- `tools/streaming/mediamtx.yml`: configurazione minima del server locale MediaMTX
- `tools/streaming/README.md`: comandi rapidi per preview RTMP/HLS e test del detector

## Model Registry

- `tools/model_registry/drive_model_sync.py`: esporta/importa modelli su Google Drive, sia via cartella sincronizzata sia via OAuth API (con latest + checksum SHA256)
- `tools/model_registry/oauth_credentials.example.json`: template del file unico locale per OAuth (`client_secrets` + `token`)