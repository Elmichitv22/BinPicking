#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path
import sys
from typing import Optional

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vision import DEFAULT_CALIBRATION_PATH, DEFAULT_CAPTURE_DIR

DEFAULT_PATTERN_COLS = 9
DEFAULT_PATTERN_ROWS = 6
DEFAULT_SQUARE_SIZE_MM = 20.0
DEFAULT_OUTPUT_PATH = DEFAULT_CALIBRATION_PATH
DEFAULT_IMAGE_DIR = DEFAULT_CAPTURE_DIR
DEFAULT_REPORT_PATH = DEFAULT_OUTPUT_PATH.parent / "last_calibration_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibra la camara desde imagenes guardadas del tablero.")
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--pattern-cols", type=int, default=DEFAULT_PATTERN_COLS)
    parser.add_argument("--pattern-rows", type=int, default=DEFAULT_PATTERN_ROWS)
    parser.add_argument("--square-size", type=float, default=DEFAULT_SQUARE_SIZE_MM)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument(
        "--allow-tangential-distortion",
        action="store_true",
        help="Permite estimar p1/p2. Por defecto se fijan a 0 para evitar soluciones irreales.",
    )
    parser.add_argument(
        "--enable-k3",
        action="store_true",
        help="Permite estimar k3. Por defecto se fija a 0 para estabilizar la calibracion.",
    )
    parser.add_argument("--show", action="store_true", help="Muestra la deteccion en cada imagen.")
    return parser.parse_args()


def build_object_points(pattern_size: tuple[int, int], square_size_mm: float) -> np.ndarray:
    pattern_cols, pattern_rows = pattern_size
    obj_points = np.zeros((pattern_cols * pattern_rows, 3), np.float32)
    grid = np.mgrid[0:pattern_cols, 0:pattern_rows].T.reshape(-1, 2)
    obj_points[:, :2] = grid * float(square_size_mm)
    return obj_points


def save_calibration(
    output_path: Path,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    image_size: tuple[int, int],
    rms_error: float,
    pattern_size: tuple[int, int],
    square_size_mm: float,
) -> None:
    payload = {
        "mtx": camera_matrix,
        "dist": dist_coeffs,
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "rms": float(rms_error),
        "pattern_cols": int(pattern_size[0]),
        "pattern_rows": int(pattern_size[1]),
        "square_size": float(square_size_mm),
    }

    suffix = output_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        fs = cv2.FileStorage(str(output_path), cv2.FILE_STORAGE_WRITE)
        if not fs.isOpened():
            raise RuntimeError(f"No se pudo abrir el archivo de salida: {output_path}")
        for key, value in payload.items():
            fs.write(key, value)
        fs.release()
        return

    if suffix == ".pkl":
        with output_path.open("wb") as handle:
            pickle.dump(payload, handle)
        return

    raise ValueError("La salida debe terminar en .yaml, .yml o .pkl")


def save_legacy_pickle_outputs(output_dir: Path, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "calibration.pkl").open("wb") as handle:
        pickle.dump((camera_matrix, dist_coeffs), handle)
    with (output_dir / "cameraMatrix.pkl").open("wb") as handle:
        pickle.dump(camera_matrix, handle)
    with (output_dir / "dist.pkl").open("wb") as handle:
        pickle.dump(dist_coeffs, handle)


def save_calibration_report(
    report_path: Path,
    image_dir: Path,
    output_path: Path,
    pattern_size: tuple[int, int],
    square_size_mm: float,
    image_paths: list[Path],
    valid_image_paths: list[Path],
    invalid_images: list[dict[str, str]],
    rms_error: float,
) -> None:
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "image_dir": str(image_dir),
        "output_path": str(output_path),
        "pattern_cols": int(pattern_size[0]),
        "pattern_rows": int(pattern_size[1]),
        "square_size_mm": float(square_size_mm),
        "total_images": len(image_paths),
        "valid_images": [str(path) for path in valid_image_paths],
        "invalid_images": invalid_images,
        "rms": float(rms_error),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def adaptive_threshold(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5,
    )


def build_detection_attempts(gray: np.ndarray) -> list[tuple[str, np.ndarray]]:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_gray = clahe.apply(gray)
    adaptive = adaptive_threshold(gray)
    return [
        ("gray", gray),
        ("equalized", cv2.equalizeHist(gray)),
        ("clahe", clahe_gray),
        ("adaptive", adaptive),
        ("inverted", cv2.bitwise_not(gray)),
        ("adaptive-inverted", cv2.bitwise_not(adaptive)),
        ("gaussian", cv2.GaussianBlur(gray, (5, 5), 0)),
        ("clahe-gaussian", cv2.GaussianBlur(clahe_gray, (5, 5), 0)),
    ]


def resize_candidate(image: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) <= 1e-6:
        return image
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)


def rescale_corners(corners: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) <= 1e-6:
        return corners
    return np.asarray(corners, dtype=np.float32) / float(scale)


def build_calibration_flags(args: argparse.Namespace) -> int:
    flags = 0
    if not args.allow_tangential_distortion:
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
    if not args.enable_k3:
        flags |= cv2.CALIB_FIX_K3
    return flags


def detect_chessboard(gray: np.ndarray, pattern_size: tuple[int, int], criteria: tuple[int, int, float]) -> tuple[bool, Optional[np.ndarray], str]:
    attempts = build_detection_attempts(gray)
    classic_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS
    sb_flags = cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY

    for scale in (1.0, 1.5):
        for method_name, candidate in attempts:
            candidate_scaled = resize_candidate(candidate, scale)
            found, corners = cv2.findChessboardCorners(candidate_scaled, pattern_size, classic_flags)
            if found:
                corners = rescale_corners(corners, scale)
                refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                return True, refined, f"classic-{method_name}-x{scale:.1f}"

    if hasattr(cv2, "findChessboardCornersSB"):
        for scale in (1.0, 1.5):
            for method_name, candidate in attempts:
                candidate_scaled = resize_candidate(candidate, scale)
                found, corners = cv2.findChessboardCornersSB(candidate_scaled, pattern_size, sb_flags)
                if found:
                    corners = rescale_corners(corners, scale)
                    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    return True, refined, f"sb-{method_name}-x{scale:.1f}"

    return False, None, "none"


def calibration_warnings(
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    image_size: tuple[int, int],
    rms_error: float,
) -> list[str]:
    width, height = image_size
    max_dim = float(max(width, height))
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])
    dist_flat = np.asarray(dist_coeffs, dtype=np.float64).ravel()

    warnings: list[str] = []

    if not np.isfinite(camera_matrix).all() or not np.isfinite(dist_flat).all():
        warnings.append("Hay valores no finitos en mtx/dist.")
        return warnings

    if fx <= 0.0 or fy <= 0.0:
        warnings.append("La distancia focal estimada es no positiva.")
    if fx > 5.0 * max_dim or fy > 5.0 * max_dim:
        warnings.append(
            f"La focal estimada es muy grande para la resolucion ({fx:.1f}, {fy:.1f} px para {width}x{height})."
        )
    if fx < 0.3 * max_dim or fy < 0.3 * max_dim:
        warnings.append(
            f"La focal estimada es muy pequena para la resolucion ({fx:.1f}, {fy:.1f} px para {width}x{height})."
        )
    if abs(cx - width / 2.0) > 0.25 * width or abs(cy - height / 2.0) > 0.25 * height:
        warnings.append(
            f"El punto principal quedo lejos del centro de imagen (cx={cx:.1f}, cy={cy:.1f})."
        )
    if min(fx, fy) > 0.0 and max(fx, fy) / min(fx, fy) > 1.2:
        warnings.append(f"Hay anisotropia alta entre fx y fy ({fx:.1f} vs {fy:.1f}).")

    k1 = float(dist_flat[0]) if len(dist_flat) > 0 else 0.0
    k2 = float(dist_flat[1]) if len(dist_flat) > 1 else 0.0
    p1 = float(dist_flat[2]) if len(dist_flat) > 2 else 0.0
    p2 = float(dist_flat[3]) if len(dist_flat) > 3 else 0.0
    k3 = float(dist_flat[4]) if len(dist_flat) > 4 else 0.0

    if abs(k1) > 1.0:
        warnings.append(f"k1 es alto ({k1:.4f}).")
    if abs(k2) > 50.0:
        warnings.append(f"k2 es muy alto ({k2:.4f}).")
    if abs(k3) > 5.0:
        warnings.append(f"k3 es alto ({k3:.4f}).")
    if abs(p1) > 0.05 or abs(p2) > 0.05:
        warnings.append(f"La distorsion tangencial parece alta (p1={p1:.4f}, p2={p2:.4f}).")
    if rms_error > 1.0:
        warnings.append(f"El RMS es alto ({rms_error:.4f} px).")

    return warnings


def main() -> int:
    args = parse_args()
    if args.pattern_cols <= 1 or args.pattern_rows <= 1:
        raise ValueError("El patron debe tener al menos 2x2 esquinas internas")
    if args.square_size <= 0:
        raise ValueError("square-size debe ser positivo")

    image_dir = Path(args.image_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.jpeg"))
    if not image_paths:
        print(f"No se encontraron imagenes en: {image_dir}")
        return 1

    pattern_size = (int(args.pattern_cols), int(args.pattern_rows))
    object_template = build_object_points(pattern_size, float(args.square_size))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    valid_image_paths: list[Path] = []
    invalid_images: list[dict[str, str]] = []
    image_size: Optional[tuple[int, int]] = None

    print(f"Leyendo imagenes desde: {image_dir}")
    print(f"Patron interno usado: {pattern_size[0]}x{pattern_size[1]}")

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"No se pudo abrir: {image_path}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_size = (gray.shape[1], gray.shape[0])
        found, corners, detector_name = detect_chessboard(gray, pattern_size, criteria)
        if not found or corners is None:
            print(f"NO -> {image_path.name} | detector={detector_name}")
            invalid_images.append({"path": str(image_path), "detector": detector_name})
            continue

        object_points.append(object_template.copy())
        image_points.append(corners)
        valid_image_paths.append(image_path)
        print(f"OK -> {image_path.name} | detector={detector_name}")

        if args.show:
            preview = image.copy()
            cv2.drawChessboardCorners(preview, pattern_size, corners, True)
            cv2.imshow("Calibracion desde imagenes", preview)
            cv2.waitKey(150)

    if args.show:
        cv2.destroyAllWindows()

    if len(image_points) < 3 or image_size is None:
        print("No hay suficientes imagenes validas para calibrar")
        return 1

    calibration_flags = build_calibration_flags(args)
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
        flags=calibration_flags,
    )
    save_calibration(
        output_path,
        camera_matrix,
        dist_coeffs,
        image_size,
        rms,
        pattern_size,
        float(args.square_size),
    )
    save_legacy_pickle_outputs(output_path.parent, camera_matrix, dist_coeffs)
    save_calibration_report(
        report_path,
        image_dir,
        output_path,
        pattern_size,
        float(args.square_size),
        image_paths,
        valid_image_paths,
        invalid_images,
        float(rms),
    )

    print("Calibracion completada")
    print(f"Imagenes validas: {len(image_points)}/{len(image_paths)}")
    print(f"RMS reprojection error: {rms:.6f}")
    print(
        "Flags usados: "
        f"ZERO_TANGENT_DIST={'ON' if not args.allow_tangential_distortion else 'OFF'} | "
        f"FIX_K3={'ON' if not args.enable_k3 else 'OFF'}"
    )
    print("Matriz de camara (mtx):")
    print(camera_matrix)
    print("Coeficientes de distorsion (dist):")
    print(dist_coeffs.ravel())
    print(f"Archivo guardado en: {output_path}")
    print(f"Pickles compatibles guardados en: {output_path.parent}")
    print(f"Reporte guardado en: {report_path}")
    print(f"Vectores calculados: rvecs={len(rvecs)}, tvecs={len(tvecs)}")

    warnings = calibration_warnings(camera_matrix, dist_coeffs, image_size, float(rms))
    if warnings:
        print("ADVERTENCIAS DE SANIDAD:")
        for warning in warnings:
            print(f"- {warning}")
        print("Sugerencia: captura menos imagenes repetidas y mueve el tablero por centro, bordes, esquinas, inclinaciones y distancias.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())