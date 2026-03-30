"""Publish a local webcam as an RTMP stream for detector testing."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

import cv2
import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start a local RTMP server and publish a webcam feed to it.",
    )
    parser.add_argument("--camera", type=int, default=2, help="OpenCV camera index to publish")
    parser.add_argument("--host", default="127.0.0.1", help="RTMP host")
    parser.add_argument("--port", type=int, default=1935, help="RTMP port")
    parser.add_argument("--app", default="", help="Optional RTMP app/path prefix")
    parser.add_argument("--stream", default="webcam2", help="RTMP stream key")
    parser.add_argument("--fps", type=float, default=20.0, help="Fallback FPS if camera metadata is missing")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="FFmpeg executable path")
    parser.add_argument("--mediamtx", default="mediamtx", help="MediaMTX executable path")
    parser.add_argument(
        "--reuse-server",
        action="store_true",
        help="Riusa un listener RTMP gia esistente invece di riavviare MediaMTX locale",
    )
    parser.add_argument(
        "--server-start-timeout",
        type=float,
        default=8.0,
        help="Seconds to wait for the local RTMP server to start",
    )
    return parser.parse_args()


def ensure_executable(name_or_path: str) -> str:
    resolved = shutil.which(name_or_path)
    if resolved is None:
        raise FileNotFoundError(f"Executable non trovato: {name_or_path}")
    return resolved


def is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def wait_for_port(host: str, port: int, timeout_seconds: float) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if is_port_open(host, port):
            return True
        time.sleep(0.2)
    return False


def wait_for_port_to_close(host: str, port: int, timeout_seconds: float) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if not is_port_open(host, port):
            return True
        time.sleep(0.2)
    return False


def find_listening_pid(port: int) -> int | None:
    try:
        result = subprocess.run(
            ["netstat", "-ano", "-p", "tcp"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    port_suffix = f":{port}"
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line.startswith("TCP"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        local_address, state, pid_text = parts[1], parts[3], parts[4]
        if state.upper() != "LISTENING":
            continue
        if not local_address.endswith(port_suffix):
            continue
        if pid_text.isdigit():
            return int(pid_text)
    return None


def get_process_name(pid: int) -> str | None:
    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    line = result.stdout.strip().splitlines()
    if not line:
        return None
    first = line[0].strip()
    if not first or first.startswith("INFO:"):
        return None
    parts = [part.strip('"') for part in first.split('","')]
    return parts[0].lower() if parts else None


def terminate_process(pid: int) -> None:
    subprocess.run(
        ["taskkill", "/PID", str(pid), "/T", "/F"],
        check=True,
        capture_output=True,
        text=True,
    )


def ensure_local_mediamtx(
    host: str,
    port: int,
    mediamtx_executable: str,
    mediamtx_config_path: Path,
    reuse_server: bool,
    timeout_seconds: float,
) -> subprocess.Popen[bytes] | None:
    if not is_port_open(host, port):
        print("Avvio MediaMTX...")
        process = subprocess.Popen([mediamtx_executable, str(mediamtx_config_path)])
        if not wait_for_port(host, port, timeout_seconds):
            raise RuntimeError("MediaMTX non ha aperto la porta RTMP entro il timeout previsto")
        return process

    if reuse_server:
        print(f"RTMP server gia in ascolto su {host}:{port}")
        return None

    pid = find_listening_pid(port)
    if pid is None:
        raise RuntimeError(
            f"La porta {port} e' occupata e non sono riuscito a identificare il processo. "
            "Usa --reuse-server oppure libera la porta."
        )

    process_name = get_process_name(pid)
    if process_name not in {"mediamtx.exe", "rtsp-simple-server.exe"}:
        raise RuntimeError(
            f"La porta {port} e' gia usata da PID {pid} ({process_name or 'processo sconosciuto'}). "
            "Usa --port diverso o libera la porta."
        )

    print(f"Riavvio il server locale {process_name} su porta {port} per applicare la config del progetto...")
    terminate_process(pid)
    if not wait_for_port_to_close(host, port, timeout_seconds):
        raise RuntimeError("La porta RTMP non si e' liberata dopo l'arresto del server esistente")

    process = subprocess.Popen([mediamtx_executable, str(mediamtx_config_path)])
    if not wait_for_port(host, port, timeout_seconds):
        raise RuntimeError("MediaMTX non ha aperto la porta RTMP entro il timeout previsto")
    return process


def open_camera_capture(camera_id: int) -> cv2.VideoCapture:
    if os.name == "nt":
        cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def probe_camera(cap: cv2.VideoCapture, fallback_fps: float) -> tuple[tuple[int, int], float, np.ndarray]:
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError("Impossibile leggere il primo frame dalla webcam selezionata")

    height, width = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1 or fps > 120:
        fps = fallback_fps
    return (width, height), float(fps), frame


def build_ffmpeg_command(ffmpeg_executable: str, size: tuple[int, int], fps: float, stream_url: str) -> list[str]:
    width, height = size
    return [
        ffmpeg_executable,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-re",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.2f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-tune",
        "zerolatency",
        "-pix_fmt",
        "yuv420p",
        "-f",
        "flv",
        stream_url,
    ]


def build_stream_url(host: str, port: int, app: str, stream: str) -> str:
    app = app.strip("/")
    stream = stream.strip("/")
    if app:
        return f"rtmp://{host}:{port}/{app}/{stream}"
    return f"rtmp://{host}:{port}/{stream}"


def main() -> int:
    args = parse_args()

    ffmpeg_executable = ensure_executable(args.ffmpeg)
    mediamtx_executable = ensure_executable(args.mediamtx)
    mediamtx_config_path = SCRIPT_ROOT / "mediamtx.yml"
    stream_url = build_stream_url(args.host, args.port, args.app, args.stream)

    mediamtx_process = ensure_local_mediamtx(
        host=args.host,
        port=args.port,
        mediamtx_executable=mediamtx_executable,
        mediamtx_config_path=mediamtx_config_path,
        reuse_server=args.reuse_server,
        timeout_seconds=args.server_start_timeout,
    )

    cap = open_camera_capture(args.camera)
    if not cap.isOpened():
        if mediamtx_process is not None:
            mediamtx_process.terminate()
        raise RuntimeError(f"Impossibile aprire la webcam {args.camera}")

    ffmpeg_process: subprocess.Popen[bytes] | None = None

    try:
        frame_size, fps, first_frame = probe_camera(cap, args.fps)
        print(f"Webcam {args.camera} pronta: {frame_size[0]}x{frame_size[1]} @ {fps:.2f} FPS")
        print(f"RTMP stream disponibile su {stream_url}")
        print(f"HLS viewer disponibile su http://{args.host}:8888/{args.stream}/index.m3u8")
        print(f"Test detector: python detect.py --source {stream_url}")
        print(f"Preview rapido: ffplay {stream_url}")
        print("Premi Ctrl+C qui per fermare server e publisher.")

        ffmpeg_command = build_ffmpeg_command(ffmpeg_executable, frame_size, fps, stream_url)
        ffmpeg_process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)
        if ffmpeg_process.stdin is None:
            raise RuntimeError("Impossibile aprire stdin per FFmpeg")

        try:
            ffmpeg_process.stdin.write(first_frame.tobytes())
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError("FFmpeg ha chiuso la connessione verso il server RTMP") from exc

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Frame non disponibile dalla webcam, interruzione publisher")
                break

            if ffmpeg_process.poll() is not None:
                raise RuntimeError(f"FFmpeg terminato con codice {ffmpeg_process.returncode}")

            try:
                ffmpeg_process.stdin.write(frame.tobytes())
            except (BrokenPipeError, OSError) as exc:
                raise RuntimeError("Publisher RTMP disconnesso durante lo streaming") from exc

    except KeyboardInterrupt:
        print("\nInterruzione richiesta, arresto stream...")
    finally:
        cap.release()

        if ffmpeg_process is not None:
            if ffmpeg_process.stdin is not None and not ffmpeg_process.stdin.closed:
                ffmpeg_process.stdin.close()
            ffmpeg_process.wait(timeout=5)

        if mediamtx_process is not None:
            mediamtx_process.terminate()
            try:
                mediamtx_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                mediamtx_process.kill()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())