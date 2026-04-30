#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vision import (
    DEFAULT_CAPTURE_DIR,
    DEFAULT_CAMERA_FPS,
    DEFAULT_CAMERA_HEIGHT,
    DEFAULT_CAMERA_INDEX,
    DEFAULT_CAMERA_WIDTH,
    DEFAULT_CAMERA_ZOOM,
    apply_digital_zoom,
    highgui_available,
    open_video_source,
)


DEFAULT_CALIBRATION_CAPTURE_ZOOM = 1.0
DEFAULT_PATTERN_COLS = 9
DEFAULT_PATTERN_ROWS = 6
DEFAULT_EXPOSURE_TIME_US = 12000
DEFAULT_ANALOGUE_GAIN = 1.0
DEFAULT_AWB_RED_GAIN = 1.0
DEFAULT_AWB_BLUE_GAIN = 1.0


def parse_af_mode(value: str) -> int:
    mapping = {
        "manual": 0,
        "auto": 1,
        "continuous": 2,
    }
    key = value.strip().lower()
    if key not in mapping:
        raise argparse.ArgumentTypeError("af-mode debe ser manual, auto o continuous")
    return mapping[key]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Captura imagenes de tablero para calibracion de camara.")
    parser.add_argument("--source", default="", help="Ruta a video o stream. Vacio usa camara en vivo.")
    parser.add_argument("--width", type=int, default=DEFAULT_CAMERA_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_CAMERA_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_CAMERA_FPS)
    parser.add_argument("--camera-index", type=int, default=DEFAULT_CAMERA_INDEX)
    parser.add_argument("--enable-autofocus", action="store_true")
    parser.add_argument("--zoom", type=float, default=DEFAULT_CALIBRATION_CAPTURE_ZOOM)
    parser.add_argument("--pattern-cols", type=int, default=DEFAULT_PATTERN_COLS)
    parser.add_argument("--pattern-rows", type=int, default=DEFAULT_PATTERN_ROWS)
    parser.add_argument("--af-mode", type=parse_af_mode, default=0)
    parser.add_argument("--exposure-time", type=int, default=DEFAULT_EXPOSURE_TIME_US)
    parser.add_argument("--analogue-gain", type=float, default=DEFAULT_ANALOGUE_GAIN)
    parser.add_argument("--lens-position", type=float, default=None)
    parser.add_argument("--awb-red-gain", type=float, default=DEFAULT_AWB_RED_GAIN)
    parser.add_argument("--awb-blue-gain", type=float, default=DEFAULT_AWB_BLUE_GAIN)
    parser.add_argument("--output-dir", default=str(DEFAULT_CAPTURE_DIR))
    return parser.parse_args()


def draw_overlay(
    frame,
    output_dir: Path,
    capture_count: int,
    zoom: float,
    warnings: list[str],
    board_coverage: Optional[float],
) -> None:
    lines = [
        f"Directorio: {output_dir}",
        f"Capturas guardadas: {capture_count}",
        f"Zoom digital: x{zoom:.2f}",
        (
            f"Tablero: {board_coverage * 100.0:.0f}% del encuadre"
            if board_coverage is not None
            else "Tablero: no detectado"
        ),
        "Teclas: s=guardar | q=salir",
    ]
    lines.extend(f"[WARN] {message}" for message in warnings)

    panel_height = 18 + len(lines) * 24
    cv2.rectangle(frame, (12, 12), (1160, panel_height), (20, 20, 20), -1)
    for index, line in enumerate(lines):
        y = 36 + index * 24
        color = (220, 220, 220)
        if line.startswith("[WARN]"):
            color = (0, 215, 255)
        cv2.putText(frame, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)


def build_picamera2_controls(args: argparse.Namespace) -> dict[str, object]:
    controls: dict[str, object] = {
        "AeEnable": False,
        "AwbEnable": False,
        "ExposureTime": int(args.exposure_time),
        "AnalogueGain": float(args.analogue_gain),
        "ColourGains": (float(args.awb_red_gain), float(args.awb_blue_gain)),
    }

    if args.lens_position is not None:
        controls["AfMode"] = 0
        controls["LensPosition"] = float(args.lens_position)
    elif not args.enable_autofocus:
        controls["AfMode"] = int(args.af_mode)
    return controls


def detect_checkerboard_coverage(gray, pattern_size: tuple[int, int]) -> Optional[float]:
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not found and hasattr(cv2, "findChessboardCornersSB"):
        sb_flags = cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY
        found, corners = cv2.findChessboardCornersSB(gray, pattern_size, sb_flags)
    if not found or corners is None:
        return None

    x, y, w, h = cv2.boundingRect(corners)
    frame_h, frame_w = gray.shape[:2]
    return max(float(w) / float(frame_w), float(h) / float(frame_h))


def next_capture_index(output_dir: Path) -> int:
    existing = sorted(output_dir.glob("img_*.png"))
    if not existing:
        return 1
    last_name = existing[-1].stem
    try:
        return int(last_name.split("_")[-1]) + 1
    except ValueError:
        return len(existing) + 1


def main() -> int:
    args = parse_args()
    if args.zoom < 1.0:
        raise ValueError("zoom debe ser mayor o igual a 1.0")
    if args.pattern_cols <= 1 or args.pattern_rows <= 1:
        raise ValueError("El patron debe tener al menos 2x2 esquinas internas")
    if args.exposure_time <= 0:
        raise ValueError("exposure-time debe ser positivo")
    if args.analogue_gain <= 0:
        raise ValueError("analogue-gain debe ser positivo")
    if not highgui_available():
        raise RuntimeError("OpenCV HighGUI no esta disponible")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    picamera2_controls = build_picamera2_controls(args) if not args.source else None

    camera, backend, diagnostics = open_video_source(
        args.source,
        args.width,
        args.height,
        args.fps,
        args.camera_index,
        args.enable_autofocus,
        picamera2_controls,
    )

    if camera is None:
        raise RuntimeError(f"No se pudo abrir la fuente de video: {diagnostics}")

    print(f"Fuente de video: {backend}")
    if diagnostics:
        print(f"Diagnostico: {diagnostics}")
    print(f"Guardando capturas en: {output_dir}")
    print("Este directorio es el que usa calibrar_desde_imagenes.py y previsualizar_undistort.py por defecto.")
    print(f"Zoom de captura: x{args.zoom:.2f}")
    if abs(float(args.zoom) - 1.0) > 1e-6:
        print("[WARN] Para calibracion intrinseca se recomienda capturar con zoom digital x1.00.")
    if backend == "picamera2" and picamera2_controls is not None:
        af_mode_desc = picamera2_controls.get("AfMode", "auto-lock")
        print("Controles Picamera2 manuales:")
        print(
            "  "
            f"AeEnable={picamera2_controls['AeEnable']} "
            f"AwbEnable={picamera2_controls['AwbEnable']} "
            f"ExposureTime={picamera2_controls['ExposureTime']}us "
            f"AnalogueGain={picamera2_controls['AnalogueGain']:.2f} "
            f"ColourGains={picamera2_controls['ColourGains']} "
            f"AfMode={af_mode_desc}"
        )
        if "LensPosition" in picamera2_controls:
            print(f"  LensPosition={picamera2_controls['LensPosition']}")

    capture_index = next_capture_index(output_dir)
    capture_count = len(list(output_dir.glob("img_*.png")))
    window_name = "Captura calibracion"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    pattern_size = (int(args.pattern_cols), int(args.pattern_rows))
    frame_counter = 0
    board_coverage: Optional[float] = None

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

            frame = apply_digital_zoom(frame, float(args.zoom))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            warnings: list[str] = []

            if int(gray.max()) >= 250:
                warnings.append("imagen posiblemente sobreexpuesta")

            if frame_counter % 10 == 0:
                board_coverage = detect_checkerboard_coverage(gray, pattern_size)
            frame_counter += 1

            if board_coverage is not None and board_coverage < 0.50:
                warnings.append("tablero pequeno: acerquelo para que ocupe 50%-80% de la imagen")

            preview = frame.copy()
            draw_overlay(preview, output_dir, capture_count, float(args.zoom), warnings, board_coverage)
            cv2.imshow(window_name, preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in (ord("s"), ord("S")):
                image_path = output_dir / f"img_{capture_index:04d}.png"
                if not cv2.imwrite(str(image_path), frame):
                    print(f"No se pudo guardar: {image_path}")
                    continue
                print(f"Imagen guardada: {image_path}")
                capture_index += 1
                capture_count += 1
    finally:
        if hasattr(camera, "stop"):
            camera.stop()
        elif hasattr(camera, "release"):
            camera.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())