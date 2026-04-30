#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vision import (
    DEFAULT_CAPTURE_DIR,
    DEFAULT_CALIBRATION_PATH,
    DEFAULT_CALIBRATION_DIR,
    DEFAULT_CAMERA_FPS,
    DEFAULT_CAMERA_HEIGHT,
    DEFAULT_CAMERA_INDEX,
    DEFAULT_CAMERA_WIDTH,
    DEFAULT_CAMERA_ZOOM,
    apply_digital_zoom,
    build_undistort_maps,
    highgui_available,
    load_camera_calibration,
    open_video_source,
)


DEFAULT_PREVIEW_ZOOM = 1.0
DEFAULT_IMAGES_DIR = DEFAULT_CAPTURE_DIR
DEFAULT_REPORT_PATH = DEFAULT_CALIBRATION_DIR / "last_calibration_report.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Previsualiza imagen original vs corregida con la calibracion actual usando el flujo de calibracion.")
    parser.add_argument("--image", default="", help="Ruta a una imagen para comparar.")
    parser.add_argument("--images-dir", default=str(DEFAULT_IMAGES_DIR), help="Carpeta de imagenes de calibracion a usar por defecto.")
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH), help="Reporte generado por calibrar_desde_imagenes.py.")
    parser.add_argument("--all-images", action="store_true", help="Ignora el reporte y usa todas las imagenes de calibracion disponibles.")
    parser.add_argument("--source", default="", help="Ruta a video o stream. Solo se usa con --live.")
    parser.add_argument("--live", action="store_true", help="Usa camara o fuente de video en vivo en lugar de imagenes de calibracion.")
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION_PATH))
    parser.add_argument("--width", type=int, default=DEFAULT_CAMERA_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_CAMERA_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_CAMERA_FPS)
    parser.add_argument("--camera-index", type=int, default=DEFAULT_CAMERA_INDEX)
    parser.add_argument("--enable-autofocus", action="store_true")
    parser.add_argument("--zoom", type=float, default=DEFAULT_PREVIEW_ZOOM)
    return parser.parse_args()


def draw_label(frame: np.ndarray, text: str, origin: tuple[int, int]) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (15, 15, 15), 4, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)


def compose_preview(original: np.ndarray, corrected: np.ndarray) -> np.ndarray:
    left = original.copy()
    right = corrected.copy()
    draw_label(left, "Original", (20, 34))
    draw_label(right, "Corregida", (20, 34))
    divider = np.full((left.shape[0], 8, 3), 35, dtype=np.uint8)
    return np.hstack([left, divider, right])


def list_preview_images(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        return []
    return sorted(
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_report_image_paths(report_path: Path, images_dir: Path) -> list[Path]:
    if not report_path.exists():
        return []
    try:
        with report_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return []

    report_image_dir = payload.get("image_dir")
    if not report_image_dir:
        return []
    try:
        resolved_report_dir = Path(report_image_dir).expanduser().resolve()
    except Exception:
        return []
    if resolved_report_dir != images_dir:
        return []

    image_paths: list[Path] = []
    for raw_path in payload.get("valid_images", []):
        try:
            path = Path(raw_path).expanduser().resolve()
        except Exception:
            continue
        if path.exists() and path.is_file():
            image_paths.append(path)
    return image_paths


def load_preview_image(image_path: Path, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, zoom: float) -> np.ndarray:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"No se pudo abrir la imagen: {image_path}")

    original = frame.copy()
    frame_size = (original.shape[1], original.shape[0])
    map1, map2 = build_undistort_maps(camera_matrix, dist_coeffs, frame_size)
    undistorted = cv2.remap(original, map1, map2, interpolation=cv2.INTER_LINEAR)
    corrected = apply_digital_zoom(undistorted, float(zoom))
    return compose_preview(original, corrected)


def preview_images(
    image_paths: list[Path],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    zoom: float,
    calibration_path: Path,
    source_label: str,
) -> int:
    if not image_paths:
        raise RuntimeError("No se encontraron imagenes para previsualizar")

    index = 0
    window_name = "Undistort preview"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print(f"Calibracion cargada: {calibration_path}")
    print(f"Origen de imagenes: {source_label}")
    print(f"Imagenes disponibles: {len(image_paths)}")
    print(f"Zoom de previsualizacion: x{zoom:.2f}")
    print("Orden aplicado: undistort -> zoom digital")
    print("Controles: A/anterior, D/siguiente, Q o ESC para salir")

    while True:
        image_path = image_paths[index]
        preview = load_preview_image(image_path, camera_matrix, dist_coeffs, zoom)
        title = f"[{index + 1}/{len(image_paths)}] {image_path.name}"
        cv2.setWindowTitle(window_name, title)
        cv2.imshow(window_name, preview)

        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            break
        if key in (ord("d"), ord("D"), 83, 2555904):
            index = (index + 1) % len(image_paths)
            continue
        if key in (ord("a"), ord("A"), 81, 2424832):
            index = (index - 1) % len(image_paths)
            continue

    cv2.destroyAllWindows()
    return 0


def main() -> int:
    args = parse_args()
    if args.zoom < 1.0:
        raise ValueError("zoom debe ser mayor o igual a 1.0")
    if not highgui_available():
        raise RuntimeError("OpenCV HighGUI no esta disponible")

    calibration_path = Path(args.calibration).expanduser().resolve()
    camera_matrix, dist_coeffs = load_camera_calibration(calibration_path)

    if args.image:
        return preview_images(
            [Path(args.image).expanduser().resolve()],
            camera_matrix,
            dist_coeffs,
            float(args.zoom),
            calibration_path,
            "imagen explicita",
        )

    if not args.live:
        images_dir = Path(args.images_dir).expanduser().resolve()
        report_path = Path(args.report).expanduser().resolve()

        image_paths: list[Path] = []
        source_label = str(images_dir)
        if not args.all_images:
            image_paths = load_report_image_paths(report_path, images_dir)
            if image_paths:
                source_label = f"reporte validas: {report_path.name}"

        if not image_paths:
            image_paths = list_preview_images(images_dir)
            source_label = f"todas las imagenes: {images_dir}"

        if not image_paths:
            raise RuntimeError(
                f"No se encontraron imagenes en {images_dir}. Captura primero o activa --live."
            )

        return preview_images(
            image_paths,
            camera_matrix,
            dist_coeffs,
            float(args.zoom),
            calibration_path,
            source_label,
        )

    camera, backend, diagnostics = open_video_source(
        args.source,
        args.width,
        args.height,
        args.fps,
        args.camera_index,
        args.enable_autofocus,
    )
    if camera is None:
        raise RuntimeError(f"No se pudo abrir la fuente de video: {diagnostics}")

    print(f"Calibracion cargada: {calibration_path}")
    print(f"Fuente de video: {backend}")
    if diagnostics:
        print(f"Diagnostico: {diagnostics}")
    print(f"Zoom de previsualizacion: x{args.zoom:.2f}")
    print("Orden aplicado: undistort -> zoom digital")
    print("Pulsa q o ESC para salir")

    map1: Optional[np.ndarray] = None
    map2: Optional[np.ndarray] = None
    map_size: Optional[tuple[int, int]] = None
    window_name = "Undistort preview"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            if hasattr(camera, "capture_array"):
                frame = camera.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                ok, frame = camera.read()
                if not ok or frame is None:
                    print("No se pudo leer un frame de la camara")
                    break

            frame_size = (frame.shape[1], frame.shape[0])
            if map1 is None or map2 is None or map_size != frame_size:
                map1, map2 = build_undistort_maps(camera_matrix, dist_coeffs, frame_size)
                map_size = frame_size

            undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
            corrected = apply_digital_zoom(undistorted, float(args.zoom))
            preview = compose_preview(frame, corrected)
            cv2.imshow(window_name, preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        if hasattr(camera, "stop"):
            camera.stop()
        elif hasattr(camera, "release"):
            camera.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())