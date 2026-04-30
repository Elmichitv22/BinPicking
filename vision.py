#!/usr/bin/env python3
# pyright: reportMissingImports=false

from __future__ import annotations

import importlib
import math
import os
import pickle
from pathlib import Path
from typing import Any, List, Mapping, Optional, Tuple

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
BATTERY_CLASS_NAMES = {"bateria", "battery"}
DEFAULT_CAMERA_WIDTH = 1280
DEFAULT_CAMERA_HEIGHT = 720
DEFAULT_CAMERA_FPS = 30
DEFAULT_CAMERA_ZOOM = 2.0
DEFAULT_CAMERA_INDEX = -1
DEFAULT_CALIBRATION_DIR = ROOT_DIR / "calibracion"
DEFAULT_CAPTURE_DIR = DEFAULT_CALIBRATION_DIR / "calibracion_imgs"
DEFAULT_CALIBRATION_PATH = DEFAULT_CALIBRATION_DIR / "camera_calibration.yaml"

PANEL_BLUE = (200, 90, 20)
PANEL_BLUE_DARK = (150, 55, 10)
PANEL_SOFT = (220, 230, 242)
PANEL_TEXT = (70, 70, 70)
PANEL_ARROW_SOFT = (210, 140, 90)
PANEL_ARROW_SOFT_DARK = (185, 115, 70)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)


def get_object_pose(mask: np.ndarray, min_area: int = 120) -> Optional[dict]:
    """
    Calcula centro y orientacion de un objeto a partir de una mascara binaria.

    Por que no usar solo centroide:
    - El centroide (momentos) se desplaza cuando la mascara tiene rebabas,
      huecos, partes irregulares o inclinaciones fuertes.
    - En tiempo real, pequenas variaciones del contorno hacen "saltar" el centro.

    Por que minAreaRect es mas estable:
    - Ajusta una caja rotada al contorno principal y entrega centro + orientacion
      de forma consistente frente a ruido moderado.
    - Es rapido y apto para Raspberry Pi (pocas operaciones y O(N) sobre pixeles).

    Retorna None si no hay objeto valido.
    """
    try:
        if mask is None or mask.size == 0:
            return None

        # Asegura binario uint8 0/255 con una sola pasada.
        if mask.ndim == 3:
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        else:
            mask_gray = mask
        mask_u8 = np.where(mask_gray > 0, 255, 0).astype(np.uint8)

        # Limpieza ligera (kernel pequeno para bajo costo en Raspberry Pi).
        k = np.ones((3, 3), np.uint8)
        clean = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, k, iterations=1)
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, k, iterations=1)

        # Toma solo el componente principal para evitar ruido aislado.
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
        if num_labels <= 1:
            return None

        best_label = -1
        best_area = 0
        for lbl in range(1, num_labels):
            area = int(stats[lbl, cv2.CC_STAT_AREA])
            if area > best_area:
                best_area = area
                best_label = lbl

        if best_label < 0 or best_area < int(min_area):
            return None

        main_mask = np.zeros_like(clean)
        main_mask[labels == best_label] = 255

        contours, _ = cv2.findContours(main_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < float(min_area):
            return None

        # Centro + orientacion por caja rotada minima.
        (cx, cy), (w, h), angle = cv2.minAreaRect(contour)

        # Normaliza angulo al eje mayor: [0, 180).
        # Convencion OpenCV: angle en [-90, 0). Si h > w, se corrige +90.
        if h > w:
            angle = angle + 90.0
        angle = float(angle % 180.0)

        box = cv2.boxPoints(((cx, cy), (w, h), angle)).astype(np.int32)

        return {
            "center": (float(cx), float(cy)),
            "angle": angle,
            "box": box,
            "contour": contour,
        }
    except Exception:
        # No romper el pipeline en tiempo real.
        return None


def draw_pose(image: np.ndarray, pose: Optional[dict]) -> np.ndarray:
    """Dibuja contorno, caja rotada, centro y flecha de orientacion."""
    if image is None or pose is None:
        return image

    out = image.copy()
    contour = pose.get("contour", None)
    box = pose.get("box", None)
    center = pose.get("center", None)
    angle = float(pose.get("angle", 0.0))

    if contour is not None:
        cv2.drawContours(out, [contour], -1, (255, 220, 80), 2)
    if box is not None and len(box) == 4:
        cv2.polylines(out, [np.asarray(box, np.int32)], True, (0, 220, 255), 2)

    if center is None:
        return out

    cx, cy = int(round(center[0])), int(round(center[1]))
    cv2.circle(out, (cx, cy), 4, (0, 255, 0), -1)

    # Flecha en direccion del eje mayor (0..180), en coordenadas de imagen.
    if box is not None and len(box) == 4:
        box_arr = np.asarray(box, np.float32)
        side_a = float(np.linalg.norm(box_arr[1] - box_arr[0]))
        side_b = float(np.linalg.norm(box_arr[2] - box_arr[1]))
        arrow_len = int(max(20.0, 0.45 * max(side_a, side_b)))
    else:
        arrow_len = 30

    theta = math.radians(angle)
    ex = int(round(cx + arrow_len * math.cos(theta)))
    ey = int(round(cy + arrow_len * math.sin(theta)))
    cv2.arrowedLine(out, (cx, cy), (ex, ey), (0, 0, 255), 2, tipLength=0.30)
    cv2.putText(out, f"{angle:.1f} deg", (cx + 6, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return out


def default_model_path() -> Path:
    """Busca best.pt del trainN mas reciente; si no existe usa yolov8n-seg.pt."""
    runs_dir = ROOT_DIR / "runs" / "segment"
    if runs_dir.exists():
        train_dirs = sorted(
            [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("train")],
            key=lambda p: int(p.name[5:]) if len(p.name) > 5 and p.name[5:].isdigit() else 0,
            reverse=True,
        )
        for train_dir in train_dirs:
            best_pt = train_dir / "weights" / "best.pt"
            if best_pt.exists():
                return best_pt
    return ROOT_DIR / "yolov8n-seg.pt"


def highgui_available() -> bool:
    if not os.getenv("DISPLAY") and not os.getenv("WAYLAND_DISPLAY"):
        return False
    test_name = "_highgui_test_"
    try:
        cv2.namedWindow(test_name, cv2.WINDOW_NORMAL)
        cv2.imshow(test_name, np.zeros((1, 1, 3), dtype=np.uint8))
        cv2.waitKey(1)
        cv2.destroyWindow(test_name)
        return True
    except cv2.error:
        return False


def prepare_display_window(window_name: str, width: Optional[int] = None, height: Optional[int] = None) -> bool:
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        if width is not None and height is not None:
            cv2.resizeWindow(window_name, int(width), int(height))
        try:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        except Exception:
            pass
        cv2.waitKey(1)
        return True
    except cv2.error:
        return False


def apply_digital_zoom(frame: np.ndarray, zoom: float) -> np.ndarray:
    if zoom <= 1.0:
        return frame
    h, w = frame.shape[:2]
    crop_w = max(2, int(round(w / zoom)))
    crop_h = max(2, int(round(h / zoom)))
    x1 = max(0, (w - crop_w) // 2)
    y1 = max(0, (h - crop_h) // 2)
    x2 = min(w, x1 + crop_w)
    y2 = min(h, y1 + crop_h)
    cropped = frame[y1:y2, x1:x2]
    if cropped.size == 0:
        return frame
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def load_camera_calibration(calibration_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    suffix = calibration_path.suffix.lower()

    if suffix in {".yaml", ".yml"}:
        fs = cv2.FileStorage(str(calibration_path), cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise RuntimeError(f"No se pudo abrir calibracion: {calibration_path}")
        camera_matrix = fs.getNode("mtx").mat()
        dist_coeffs = fs.getNode("dist").mat()
        fs.release()
    elif suffix == ".pkl":
        with calibration_path.open("rb") as handle:
            payload = pickle.load(handle)
        camera_matrix = payload.get("mtx")
        dist_coeffs = payload.get("dist")
    else:
        raise ValueError("La calibracion debe estar en .yaml, .yml o .pkl")

    if camera_matrix is None or dist_coeffs is None:
        raise ValueError(f"El archivo de calibracion no contiene 'mtx' y 'dist': {calibration_path}")

    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64)
    return camera_matrix, dist_coeffs


def build_undistort_maps(
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    frame_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, frame_size, 0.0)
    return cv2.initUndistortRectifyMap(
        camera_matrix,
        dist_coeffs,
        None,
        new_camera_matrix,
        frame_size,
        cv2.CV_16SC2,
    )


def open_picamera2(
    width: int,
    height: int,
    fps: int,
    enable_autofocus: bool,
    camera_controls: Optional[Mapping[str, Any]] = None,
):
    try:
        module = importlib.import_module("picamera2")
        Picamera2 = getattr(module, "Picamera2")
    except Exception as exc:
        return None, None, f"picamera2 no disponible: {exc}"

    try:
        picam2 = Picamera2()
        controls = {"FrameDurationLimits": (int(1e6 / fps), int(1e6 / fps))}
        if camera_controls:
            controls.update({key: value for key, value in camera_controls.items() if value is not None})
        if not enable_autofocus and "AfMode" not in controls:
            controls.update({"AfMode": 0, "AfTrigger": 0})
        config = picam2.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"},
            controls=controls,
        )
        picam2.configure(config)
        picam2.start()
        try:
            picam2.set_controls(controls)
        except Exception:
            pass
        if "AfMode" in controls:
            try:
                picam2.set_controls({"AfMode": controls["AfMode"]})
            except Exception:
                pass
        elif enable_autofocus:
            try:
                picam2.set_controls({"AfMode": 1, "AfTrigger": 0})
                picam2.autofocus_cycle()
                picam2.set_controls({"AfMode": 0})
            except Exception:
                pass
        else:
            try:
                picam2.set_controls({"AfMode": 0, "AfTrigger": 0})
            except Exception:
                pass
        frame = picam2.capture_array()
        if frame is None or frame.size == 0:
            picam2.stop()
            return None, None, "picamera2 inicio pero no devolvio frames"
        return picam2, "picamera2", "picamera2 OK"
    except Exception as exc:
        return None, None, f"error al iniciar picamera2: {exc}"


def _try_cap(cap: cv2.VideoCapture, width: int, height: int, fps: int) -> bool:
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    try:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    except Exception:
        pass
    for _ in range(10):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            return True
    return False


def open_opencv_camera(width: int, height: int, fps: int, camera_index: int):
    diagnostics: List[str] = []
    candidates = [camera_index] if camera_index >= 0 else [0, 1, 2]

    for idx in candidates:
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            diagnostics.append(f"opencv v4l2 index {idx}: no abre")
            cap.release()
            continue
        if _try_cap(cap, width, height, fps):
            return cap, f"opencv-v4l2(index={idx})", diagnostics
        diagnostics.append(f"opencv v4l2 index {idx}: abre pero no entrega frames")
        cap.release()

    gst = (
        "libcamerasrc ! "
        f"video/x-raw,width={width},height={height},framerate={fps}/1 ! "
        "videoconvert ! appsink drop=true max-buffers=1"
    )
    cap_gst = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    if not cap_gst.isOpened():
        diagnostics.append("opencv gstreamer/libcamerasrc: no abre")
        cap_gst.release()
        return None, None, diagnostics
    if _try_cap(cap_gst, width, height, fps):
        return cap_gst, "opencv-gstreamer(libcamerasrc)", diagnostics
    diagnostics.append("opencv gstreamer/libcamerasrc: abre pero no entrega frames")
    cap_gst.release()
    return None, None, diagnostics


def open_video_source(
    source: str,
    width: int,
    height: int,
    fps: int,
    camera_index: int,
    enable_autofocus: bool,
    camera_controls: Optional[Mapping[str, Any]] = None,
):
    if source:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            return None, None, f"No se pudo abrir archivo: {source}"
        return cap, f"file({source})", "OK"

    camera, backend, diag = open_picamera2(width, height, fps, enable_autofocus, camera_controls)
    if camera is not None:
        return camera, backend, diag
    return open_opencv_camera(width, height, fps, camera_index)


def _major_axis_from_points(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    pts = np.asarray(pts, dtype=np.float32)
    if len(pts) < 3:
        center = np.mean(pts, axis=0) if len(pts) else np.zeros(2, np.float32)
        return center, np.array([1.0, 0.0], np.float32), 0.0

    rect = cv2.minAreaRect(pts.reshape(-1, 1, 2))
    box = cv2.boxPoints(rect)
    edges = [box[(i + 1) % 4] - box[i] for i in range(4)]
    lengths = [float(np.linalg.norm(e)) for e in edges]
    major = edges[int(np.argmax(lengths))]
    norm = float(np.linalg.norm(major)) + 1e-9
    major_unit = (major / norm).astype(np.float32)

    center = np.mean(pts, axis=0)
    ang = math.degrees(math.atan2(float(major_unit[1]), float(major_unit[0])))
    if ang < 0:
        ang += 180.0
    if ang >= 180.0:
        ang -= 180.0
    return center, major_unit, ang


def _polygon_centroid(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float32)
    if len(pts) < 3:
        return np.mean(pts, axis=0) if len(pts) else np.zeros(2, np.float32)
    m = cv2.moments(pts.reshape(-1, 1, 2))
    if abs(float(m.get("m00", 0.0))) > 1e-6:
        return np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]], np.float32)
    return np.mean(pts, axis=0)


def _edge_roughness_local(end_pts_local: np.ndarray, side: int) -> float:
    if len(end_pts_local) < 8:
        return 0.0
    y_vals = end_pts_local[:, 1]
    x_vals = end_pts_local[:, 0]
    y_min, y_max = float(np.min(y_vals)), float(np.max(y_vals))
    if y_max - y_min < 1e-6:
        return 0.0

    bins = np.linspace(y_min, y_max, 10)
    samples: List[float] = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_vals >= lo) & (y_vals <= hi if i == len(bins) - 2 else y_vals < hi)
        if not np.any(mask):
            continue
        x_s = x_vals[mask]
        samples.append(float(np.max(x_s)) if side > 0 else float(-np.min(x_s)))
    if len(samples) < 3:
        return 0.0
    return float(np.std(np.asarray(samples, np.float32))) * float(len(samples)) / float(len(bins) - 1)


def _score_end_local(end_pts_local: np.ndarray, side: int) -> float:
    if len(end_pts_local) == 0:
        return -1.0
    x = end_pts_local[:, 0]
    y = end_pts_local[:, 1]
    width = float(np.percentile(np.abs(y), 90)) if len(y) > 0 else 0.0
    protrusion = float(np.percentile(side * x, 95)) if len(x) > 0 else 0.0
    roughness = _edge_roughness_local(end_pts_local, side)
    return 0.45 * protrusion + 0.35 * roughness + 0.20 * width


def _score_end_global(end_pts: np.ndarray, center: np.ndarray, perp_unit: np.ndarray) -> float:
    if len(end_pts) == 0:
        return -1.0
    rel = end_pts - center
    perp = np.abs(rel @ perp_unit)
    width = float(np.percentile(perp, 90)) if len(perp) > 0 else 0.0
    radial = np.linalg.norm(rel, axis=1)
    protrusion = float(np.percentile(radial, 90)) if len(radial) > 0 else 0.0
    return 0.7 * width + 0.3 * protrusion


def _skewness(x: np.ndarray) -> float:
    if len(x) < 5:
        return 0.0
    mu = float(np.mean(x))
    sigma = float(np.std(x)) + 1e-9
    return float(np.mean(((x - mu) / sigma) ** 3))


def _compass_from_image_vector(direction_vec: np.ndarray) -> float:
    img_angle = math.degrees(math.atan2(float(direction_vec[1]), float(direction_vec[0])))
    if img_angle < 0.0:
        img_angle += 360.0

    compass_base = (img_angle + 90.0) % 360.0
    return (360.0 - compass_base) % 360.0


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _north_angle_from_tabs(
    pts_xy: np.ndarray,
    center_xy: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float, np.ndarray, np.ndarray, float]:
    pts = np.asarray(pts_xy, np.float32)
    center, major_unit, axis_angle = _major_axis_from_points(pts)
    if center_xy is not None:
        center = np.asarray(center_xy, np.float32)
    else:
        center = _polygon_centroid(pts)

    body_center = np.asarray(_core_center_from_mask(pts), np.float32)
    perp_unit = np.array([-major_unit[1], major_unit[0]], np.float32)

    rel_body = pts - body_center
    x_body = rel_body @ major_unit
    y_body = rel_body @ perp_unit

    pos_extent = float(np.percentile(x_body, 98)) if len(x_body) > 0 else 0.0
    neg_extent = float(-np.percentile(x_body, 2)) if len(x_body) > 0 else 0.0
    tab_side = 1.0 if pos_extent >= neg_extent else -1.0

    dominant_extent = max(pos_extent, neg_extent, 1.0)
    end_band = max(2.0, 0.18 * dominant_extent)
    if tab_side > 0:
        tab_mask = x_body >= (pos_extent - end_band)
    else:
        tab_mask = x_body <= (-neg_extent + end_band)

    if not np.any(tab_mask):
        north_vec = major_unit if tab_side > 0 else -major_unit
        tab_centroid = center + north_vec
    else:
        tab_points = pts[tab_mask]
        tab_centroid = np.mean(tab_points, axis=0)
        north_vec = tab_centroid - center
        vec_norm = float(np.linalg.norm(north_vec))
        if vec_norm <= 1e-6:
            north_vec = major_unit if tab_side > 0 else -major_unit
        else:
            north_vec = (north_vec / vec_norm).astype(np.float32)

    area = abs(float(cv2.contourArea(pts.reshape(-1, 1, 2))))
    hull = cv2.convexHull(pts.reshape(-1, 1, 2))
    hull_area = abs(float(cv2.contourArea(hull))) if hull is not None else 0.0
    solidity = area / max(hull_area, 1e-6)

    rect = cv2.minAreaRect(pts.reshape(-1, 1, 2))
    rect_w, rect_h = float(rect[1][0]), float(rect[1][1])
    rect_area = max(rect_w * rect_h, 1e-6)
    fill_ratio = area / rect_area

    dominant_extent = max(pos_extent, neg_extent, 1e-6)
    opposite_extent = neg_extent if tab_side > 0 else pos_extent
    tab_prominence = abs(pos_extent - neg_extent) / dominant_extent
    tab_balance = (dominant_extent - opposite_extent) / dominant_extent

    tab_width = float(np.percentile(np.abs(y_body[tab_mask]), 85)) if np.any(tab_mask) else 0.0
    body_width = float(np.percentile(np.abs(y_body), 85)) if len(y_body) > 0 else 0.0
    tab_width_ratio = tab_width / max(body_width, 1e-6)
    tab_point_ratio = float(np.count_nonzero(tab_mask)) / max(float(len(pts)), 1.0)

    center_to_tab = float(np.linalg.norm(np.asarray(tab_centroid, np.float32) - center))
    center_to_tab_ratio = center_to_tab / max(dominant_extent, 1e-6)

    solidity_score = _clamp01((solidity - 0.72) / 0.20)
    fill_score = _clamp01((fill_ratio - 0.45) / 0.30)
    prominence_score = _clamp01((tab_prominence - 0.08) / 0.24)
    width_score = _clamp01((tab_width_ratio - 0.18) / 0.28)
    points_score = _clamp01((tab_point_ratio - 0.06) / 0.18)
    vector_score = _clamp01((center_to_tab_ratio - 0.55) / 0.45)

    confidence = (
        0.26 * prominence_score
        + 0.18 * width_score
        + 0.16 * points_score
        + 0.18 * vector_score
        + 0.12 * solidity_score
        + 0.10 * fill_score
    )
    confidence *= _clamp01((tab_balance - 0.05) / 0.35)

    compass = _compass_from_image_vector(north_vec)
    return compass, axis_angle, center, north_vec, _clamp01(confidence)


def _stabilize_180(current: float, previous: Optional[float]) -> float:
    if previous is None:
        return current
    c1 = current % 360.0
    c2 = (current + 180.0) % 360.0
    d1 = abs(((c1 - previous + 180.0) % 360.0) - 180.0)
    d2 = abs(((c2 - previous + 180.0) % 360.0) - 180.0)
    return c1 if d1 <= d2 else c2


def _ema_circular(current: float, previous: Optional[float], alpha: float) -> float:
    if previous is None:
        return current
    delta = ((current - previous + 180.0) % 360.0) - 180.0
    return (previous + alpha * delta) % 360.0


def _instance_area(result, idx: int) -> float:
    boxes = result.boxes
    masks = result.masks
    if masks is not None and masks.xy is not None and idx < len(masks.xy):
        pts = masks.xy[idx]
        if pts is not None and len(pts) >= 3:
            return abs(float(cv2.contourArea(np.asarray(pts, np.float32).reshape(-1, 1, 2))))
    if boxes is not None:
        x1, y1, x2, y2 = boxes.xyxy[idx].tolist()
        return max(0.0, float((x2 - x1) * (y2 - y1)))
    return 0.0


def _force_point_inside_polygon(point_xy: Tuple[float, float], pts_xy: np.ndarray) -> Tuple[float, float]:
    pts = np.asarray(pts_xy, np.float32).reshape(-1, 1, 2)
    px, py = float(point_xy[0]), float(point_xy[1])

    if cv2.pointPolygonTest(pts, (px, py), False) >= 0:
        return px, py

    c = _polygon_centroid(np.asarray(pts_xy, np.float32))
    cx, cy = float(c[0]), float(c[1])
    for a in (0.85, 0.70, 0.55, 0.40, 0.25):
        tx = a * px + (1.0 - a) * cx
        ty = a * py + (1.0 - a) * cy
        if cv2.pointPolygonTest(pts, (tx, ty), False) >= 0:
            return tx, ty

    x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
    w = max(2, int(w + 3))
    h = max(2, int(h + 3))
    mask = np.zeros((h, w), dtype=np.uint8)
    shifted = pts.copy()
    shifted[:, 0, 0] = shifted[:, 0, 0] - x + 1
    shifted[:, 0, 1] = shifted[:, 0, 1] - y + 1
    cv2.fillPoly(mask, [shifted.astype(np.int32)], 255)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, _, _, max_loc = cv2.minMaxLoc(dist)
    return float(max_loc[0] + x - 1), float(max_loc[1] + y - 1)


def _polygon_to_local_mask(pts_xy: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Rasteriza un poligono a mascara local y devuelve offset (x0, y0)."""
    pts = np.asarray(pts_xy, np.float32).reshape(-1, 2)
    x, y, w, h = cv2.boundingRect(pts.astype(np.int32).reshape(-1, 1, 2))
    w = max(2, int(w + 4))
    h = max(2, int(h + 4))
    x0 = float(x - 2)
    y0 = float(y - 2)

    local = np.zeros((h, w), dtype=np.uint8)
    shifted = np.empty((len(pts), 1, 2), dtype=np.int32)
    shifted[:, 0, 0] = np.round(pts[:, 0] - x0).astype(np.int32)
    shifted[:, 0, 1] = np.round(pts[:, 1] - y0).astype(np.int32)
    cv2.fillPoly(local, [shifted], 255)
    return local, x0, y0


def _core_center_from_mask(pts_xy: np.ndarray) -> Tuple[float, float]:
    """Centro por nucleo interior de la mascara (insensible a pestañas y bordes)."""
    mask, x0, y0 = _polygon_to_local_mask(pts_xy)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dmax = float(np.max(dist)) if dist.size else 0.0
    if dmax <= 1e-6:
        c = _polygon_centroid(np.asarray(pts_xy, np.float32))
        return float(c[0]), float(c[1])

    # Nucleo interno: pixeles con alta distancia al borde.
    core = dist >= (0.55 * dmax)
    ys, xs = np.where(core)
    if len(xs) < 4:
        _, _, _, max_loc = cv2.minMaxLoc(dist)
        return float(x0 + max_loc[0]), float(y0 + max_loc[1])

    cx = float(x0 + np.mean(xs))
    cy = float(y0 + np.mean(ys))
    return cx, cy


def _robust_mask_center(pts_xy: np.ndarray) -> Tuple[float, float]:
    pts = np.asarray(pts_xy, np.float32)
    if len(pts) < 5:
        c = _polygon_centroid(pts)
        return float(c[0]), float(c[1])

    c_poly = _polygon_centroid(pts)
    cx_core, cy_core = _core_center_from_mask(pts)

    # Mezcla leve con centro poligonal para estabilidad subpixel.
    cx = 0.78 * float(cx_core) + 0.22 * float(c_poly[0])
    cy = 0.78 * float(cy_core) + 0.22 * float(c_poly[1])
    return _force_point_inside_polygon((cx, cy), pts)


def _battery_body_center(pts_xy: np.ndarray) -> Tuple[float, float]:
    pts = np.asarray(pts_xy, np.float32)
    if len(pts) < 5:
        return _robust_mask_center(pts)

    cx_core, cy_core = _core_center_from_mask(pts)
    return _force_point_inside_polygon((float(cx_core), float(cy_core)), pts)


def select_table_indices(result) -> List[int]:
    boxes = result.boxes
    if boxes is None:
        return []
    out = []
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item()) if boxes.cls is not None else -1
        cls_name = result.names.get(cls_id, str(cls_id)) if hasattr(result, "names") else str(cls_id)
        if str(cls_name).lower() == "mesa":
            out.append(i)
    return out


def list_non_table_indices(result) -> List[int]:
    boxes = result.boxes
    if boxes is None:
        return []
    out = []
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item()) if boxes.cls is not None else -1
        cls_name = result.names.get(cls_id, str(cls_id)) if hasattr(result, "names") else str(cls_id)
        if str(cls_name).lower() != "mesa":
            out.append(i)
    return out


def select_primary_table_index(result) -> Optional[int]:
    indices = select_table_indices(result)
    if not indices:
        return None
    return max(indices, key=lambda i: _instance_area(result, i))


def extract_instances_info(result, indices: Optional[List[int]] = None) -> List[dict]:
    info: List[dict] = []
    boxes = result.boxes
    masks = result.masks
    if boxes is None:
        return info

    n = len(boxes)
    iter_idx = range(n) if indices is None else [i for i in indices if 0 <= i < n]
    for i in iter_idx:
        cls_id = int(boxes.cls[i].item()) if boxes.cls is not None else -1
        conf = float(boxes.conf[i].item()) if boxes.conf is not None else 0.0
        cls_name = result.names.get(cls_id, str(cls_id)) if hasattr(result, "names") else str(cls_id)

        cx, cy = 0.0, 0.0
        axis_angle = 0.0
        compass = 0.0
        compass_confidence = 0.0

        if masks is not None and masks.xy is not None and i < len(masks.xy):
            pts = masks.xy[i]
            if pts is not None and len(pts) >= 5:
                pts_arr = np.asarray(pts, np.float32)
                cls_name_norm = str(cls_name).strip().lower()
                if cls_name_norm in BATTERY_CLASS_NAMES:
                    cx, cy = _battery_body_center(pts_arr)
                    compass, axis_angle, _, _, compass_confidence = _north_angle_from_tabs(pts_arr, center_xy=(cx, cy))
                else:
                    compass, axis_angle, _, _, compass_confidence = _north_angle_from_tabs(pts_arr)
                    cx, cy = _robust_mask_center(pts)
            else:
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        else:
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        info.append(
            {
                "det_idx": i,
                "class": cls_name,
                "conf": conf,
                "center": (cx, cy),
                "axis_angle_deg": axis_angle,
                "compass_bearing": compass,
                "compass_confidence": compass_confidence,
            }
        )

    return info


def refine_instance_pose_from_edges(
    frame_bgr: np.ndarray,
    result,
    det_idx: int,
    default_center: Tuple[float, float],
    default_compass: float,
) -> Tuple[Tuple[float, float], float]:
    masks = getattr(result, "masks", None)
    if frame_bgr is None or masks is None or masks.xy is None or det_idx >= len(masks.xy):
        return default_center, default_compass

    pts = masks.xy[det_idx]
    if pts is None or len(pts) < 5:
        return default_center, default_compass

    pts_arr = np.asarray(pts, np.float32)
    x, y, w, h = cv2.boundingRect(pts_arr.astype(np.int32).reshape(-1, 1, 2))
    pad = 4
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(frame_bgr.shape[1], x + w + pad)
    y1 = min(frame_bgr.shape[0], y + h + pad)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return default_center, default_compass

    roi = frame_bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    edges = cv2.Canny(gray, 80, 160)

    local_mask = np.zeros(edges.shape, dtype=np.uint8)
    shifted = np.round(pts_arr - np.array([x0, y0], dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(local_mask, [shifted], 255)
    edges = cv2.bitwise_and(edges, local_mask)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return default_center, default_compass

    default_cx = float(default_center[0]) - float(x0)
    default_cy = float(default_center[1]) - float(y0)

    best_rect = None
    best_score = None
    for contour in contours:
        if len(contour) < 5:
            continue
        area = float(cv2.contourArea(contour))
        if area < 25.0:
            continue
        (cx, cy), (rw, rh), angle = cv2.minAreaRect(contour)
        major = max(float(rw), float(rh))
        minor = max(1.0, min(float(rw), float(rh)))
        elongation = major / minor
        dist = math.hypot(float(cx) - default_cx, float(cy) - default_cy)
        score = area + 20.0 * elongation - 2.0 * dist
        if best_score is None or score > best_score:
            best_score = score
            best_rect = ((cx, cy), (rw, rh), angle)

    if best_rect is None:
        return default_center, default_compass

    (cx, cy), (rw, rh), angle = best_rect
    if rh > rw:
        angle += 90.0
    angle = float(angle % 180.0)

    refined_center = (float(cx) + float(x0), float(cy) + float(y0))
    return refined_center, default_compass


def compute_table_workobject_pose(result, mesa_idx: Optional[int]):
    if mesa_idx is None or result.boxes is None:
        return None

    boxes = result.boxes
    masks = result.masks

    x1, y1, x2, y2 = boxes.xyxy[mesa_idx].tolist()
    bw, bh = max(1.0, float(x2 - x1)), max(1.0, float(y2 - y1))
    axis_len = int(max(35.0, min(bw, bh) * 0.22))

    origin_pt = np.array([x1, y2], np.float32)
    x_axis = np.array([1.0, 0.0], np.float32)
    y_axis = np.array([0.0, -1.0], np.float32)

    if masks is not None and masks.xy is not None and mesa_idx < len(masks.xy):
        pts = masks.xy[mesa_idx]
        if pts is not None and len(pts) >= 3:
            pts_arr = np.asarray(pts, np.float32)
            rect = cv2.minAreaRect(pts_arr.reshape(-1, 1, 2))
            box = cv2.boxPoints(rect).astype(np.float32)

            y_sorted = np.argsort(box[:, 1])
            top_two = box[y_sorted[:2]]
            bottom_two = box[y_sorted[2:]]

            if float(bottom_two[0][0]) <= float(bottom_two[1][0]):
                origin_pt = bottom_two[0]
                bottom_right = bottom_two[1]
            else:
                origin_pt = bottom_two[1]
                bottom_right = bottom_two[0]

            top_left = top_two[0] if float(top_two[0][0]) <= float(top_two[1][0]) else top_two[1]

            x_axis = bottom_right - origin_pt
            y_axis = top_left - origin_pt
            if x_axis[0] < 0:
                x_axis = -x_axis
            if y_axis[1] > 0:
                y_axis = -y_axis

    x_axis = x_axis / (float(np.linalg.norm(x_axis)) + 1e-9)
    y_axis = y_axis / (float(np.linalg.norm(y_axis)) + 1e-9)

    cxi, cyi = int(round(float(origin_pt[0]))), int(round(float(origin_pt[1])))
    px = (int(round(cxi + axis_len * float(x_axis[0]))), int(round(cyi + axis_len * float(x_axis[1]))))
    py = (int(round(cxi + axis_len * float(y_axis[0]))), int(round(cyi + axis_len * float(y_axis[1]))))

    return (cxi, cyi), px, py


def draw_table_workobject(frame: np.ndarray, result, mesa_idx: Optional[int], fixed_pose=None) -> np.ndarray:
    pose = fixed_pose if fixed_pose is not None else compute_table_workobject_pose(result, mesa_idx)
    if pose is None:
        return frame

    (cxi, cyi), px, py = pose
    cv2.circle(frame, (cxi, cyi), 6, PANEL_SOFT, -1)
    cv2.putText(frame, "WO", (cxi + 8, cyi - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, PANEL_BLUE, 2, cv2.LINE_AA)
    cv2.arrowedLine(frame, (cxi, cyi), px, COLOR_RED, 3, tipLength=0.22)
    cv2.arrowedLine(frame, (cxi, cyi), py, COLOR_YELLOW, 3, tipLength=0.22)
    cv2.putText(frame, "X", (px[0] + 6, px[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2, cv2.LINE_AA)
    cv2.putText(frame, "Y", (py[0] + 6, py[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_YELLOW, 2, cv2.LINE_AA)
    return frame


def _draw_corner_box(img, p1, p2, color, thickness=2, corner_len=22):
    x1, y1 = p1
    x2, y2 = p2
    cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, thickness)
    cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, thickness)
    cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, thickness)
    cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, thickness)


def draw_detections(frame: np.ndarray, result, non_table_indices, mesa_indices) -> np.ndarray:
    out = frame.copy()
    boxes = result.boxes
    masks = result.masks

    if boxes is None:
        cv2.putText(out, "Sin detecciones", (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        return out

    for mi in mesa_indices:
        x1, y1, x2, y2 = boxes.xyxy[mi].tolist()
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), PANEL_BLUE_DARK, 2)
        m_conf = float(boxes.conf[mi].item()) if boxes.conf is not None else 0.0
        cv2.putText(
            out,
            f"MESA {m_conf:.2f}",
            (int(x1), max(20, int(y1) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            PANEL_BLUE_DARK,
            2,
            cv2.LINE_AA,
        )

    for ti in non_table_indices:
        cls_id = int(boxes.cls[ti].item()) if boxes.cls is not None else -1
        conf = float(boxes.conf[ti].item()) if boxes.conf is not None else 0.0
        cls_name = result.names.get(cls_id, str(cls_id)) if hasattr(result, "names") else str(cls_id)

        is_battery = str(cls_name).strip().lower() in BATTERY_CLASS_NAMES
        obj_color = PANEL_BLUE if is_battery else PANEL_BLUE_DARK

        if masks is not None and masks.xy is not None and ti < len(masks.xy):
            pts = masks.xy[ti]
            if pts is not None and len(pts) >= 3:
                cv2.polylines(out, [np.asarray(pts, np.int32).reshape(-1, 1, 2)], True, obj_color, 2)

        x1, y1, x2, y2 = boxes.xyxy[ti].tolist()
        _draw_corner_box(out, (int(x1), int(y1)), (int(x2), int(y2)), obj_color, 3, 24)
        cv2.putText(
            out,
            f"{cls_name} {conf:.2f}",
            (int(x1), max(20, int(y1) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            obj_color,
            2,
            cv2.LINE_AA,
        )

    if not non_table_indices:
        cv2.putText(out, "Sin objetivo", (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, PANEL_BLUE_DARK, 2, cv2.LINE_AA)

    return out


def _draw_compass_arrow(
    img: np.ndarray,
    cx: int,
    cy: int,
    compass_deg: float,
    length: int = 50,
    color: Tuple[int, int, int] = PANEL_ARROW_SOFT,
    thickness: int = 2,
    label: str = "N",
) -> None:
    image_angle = math.radians((270.0 - compass_deg) % 360.0)
    nx = int(round(cx + length * math.cos(image_angle)))
    ny = int(round(cy + length * math.sin(image_angle)))
    cv2.arrowedLine(img, (cx, cy), (nx, ny), color, thickness, tipLength=0.28)
    cv2.putText(img, label, (nx + 4, ny - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def draw_center_orientation(frame: np.ndarray, instances_info: List[dict]) -> np.ndarray:
    out = frame
    for obj in instances_info:
        cx, cy = obj["center"]
        cxi, cyi = int(round(cx)), int(round(cy))
        axis_angle = obj["axis_angle_deg"]
        compass = obj["compass_bearing"]
        class_name = str(obj["class"]).strip().lower()
        is_table = class_name == "mesa"
        is_battery = class_name in BATTERY_CLASS_NAMES

        cv2.circle(out, (cxi, cyi), 5, PANEL_SOFT, -1)

        if is_table:
            cv2.putText(
                out,
                f"MESA ({cxi},{cyi})",
                (cxi + 8, cyi - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                PANEL_TEXT,
                1,
                cv2.LINE_AA,
            )
            continue

        line_len = 45
        theta_img = math.radians(axis_angle)
        ex1 = int(round(cxi + line_len * math.cos(theta_img)))
        ey1 = int(round(cyi + line_len * math.sin(theta_img)))
        ex2 = int(round(cxi - line_len * math.cos(theta_img)))
        ey2 = int(round(cyi - line_len * math.sin(theta_img)))
        cv2.line(out, (ex2, ey2), (ex1, ey1), PANEL_BLUE_DARK, 2)

        if is_battery:
            _draw_compass_arrow(out, cxi, cyi, compass, length=52, color=COLOR_RED, thickness=2, label="N")
        else:
            _draw_compass_arrow(out, cxi, cyi, compass, length=52, color=PANEL_ARROW_SOFT_DARK, thickness=2)

        text = f"{obj['class']} ({cxi},{cyi}) rumbo={compass:.1f}deg"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tx, ty = cxi + 8, cyi - 8
        cv2.rectangle(out, (tx - 2, ty - th - 2), (tx + tw + 2, ty + 2), (0, 0, 0), cv2.FILLED)
        cv2.putText(out, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return out
