#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

# Controla si se muestran los prints de debug de parejas
PAIR_DEBUG_ENABLED = False

# Controla si se muestran los prints de lecturas Modbus (Function Code 03)
HMI_DEBUG_READS = False

import struct
import socket
import sys
import time
import threading
import argparse
import json
import pickle
import math
from pathlib import Path

# DirecciÃƒÂ³n IP de la HMI
HMI_IP = "192.168.250.11"  # Cambia aquÃƒÂ­ la IP real de la HMI si es diferente
from typing import Optional, Tuple
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

from robot_io import EpsonModbusClient

from vision import (
    apply_digital_zoom,
    default_model_path,
    draw_table_workobject,
    DEFAULT_CAMERA_FPS,
    DEFAULT_CAMERA_HEIGHT,
    DEFAULT_CAMERA_INDEX,
    DEFAULT_CAMERA_WIDTH,
    DEFAULT_CAMERA_ZOOM,
    extract_instances_info,
    highgui_available,
    list_non_table_indices,
    open_video_source,
    select_primary_table_index,
)

BATTERY_CLASS_NAMES = {"bateria", "battery"}
TAB_CLASS_NAMES = {"pestana", "pestania", "tab"}
TAB_BATTERY_EXPAND_PX = 20.0
TAB_MATCH_MAX_DIST_PX = 80.0
TAB_MIN_CONF = 0.30
BATTERY_MIN_CONF = 0.40
TAB_TO_BATTERY_MASK_MAX_DIST_PX = 25.0
TAB_CENTER_TO_BATTERY_CENTER_MIN_DIST_PX = 5.0
TAB_MAX_EDGE_DIST_PX = 25.0
TAB_MIN_AXIS_PROJECTION_RATIO = 0.35
TAB_MAX_PERP_DIST_RATIO = 0.45
PAIR_SCORE_TAB_MASK_WEIGHT = 5.0
PAIR_SCORE_CENTER_WEIGHT = 0.2
PAIR_SCORE_CONF_WEIGHT = 40.0
PICK_OVERLAP_MAX_IOU = 0.15
PAIR_DEBUG_PERIOD_S = 0.5
TABLE_EDGE_MARGIN_PX = 20.0
DEBUG_PAIR_LINES = False
FORCE_SELECT_NEAREST_TAB = False
TAB_FORCE_MAX_DIST_PX = 80.0
TAB_U_BIAS_DEG = 0.0
BATTERY_AXIS_U_BIAS_DEG = 270.0

START = 520 - 1
STOP = 521 - 1
PAUSE = 522 - 1
CONTINUE = 523 - 1
RESET = 524 - 1

REG_X = 32
REG_Y = 48
REG_Z = 64
REG_U = 80
REG_SPEED = 96

DEFAULT_SPEED = 40

HMI_SERVER_PORT = 1502

HMI_START = 519
HMI_STOP = 520
HMI_PAUSE = 521
HMI_CONTINUE = 522
HMI_RESET = 523

hmi_coils = {}
hmi_lock = threading.Lock()

hmi_registers = {}
hmi_registers_lock = threading.Lock()

hmi_pending_cmds = set()
hmi_pending_lock = threading.Lock()

hmi_speed_lock = threading.Lock()
last_speed_sent_to_robot = None
speed_pending = False

HMI_BUTTON_NAMES = {
    HMI_START: "START",
    HMI_STOP: "STOP",
    HMI_PAUSE: "PAUSE",
    HMI_CONTINUE: "CONTINUE",
    HMI_RESET: "RESET",
}

POSE_SCALE = 100.0
TABLE_WIDTH_MM = 209.00
TABLE_HEIGHT_MM = 209.00
CAMERA_X_OFFSET_MM = 0.00
CAMERA_Y_OFFSET_MM = 0.00
MANUAL_ROBOT_POINTS = [
    (20.0, 20.0),
    (104.5, 20.0),
    (189.0, 20.0),
    (20.0, 104.5),
    (104.5, 104.5),
    (189.0, 104.5),
    (20.0, 189.0),
    (104.5, 189.0),
    (189.0, 189.0),
]

Z_OFFSET_MM = 100.0

SEND_DELTA_MM = 0.4
SEND_DELTA_DEG = 2.0
WORDS_SEND_MIN_PERIOD_S = 0.04
NO_BATTERY_STOP_DELAY_S = 8.0
ATTEMPTS_LOG_PATH = Path(__file__).resolve().parent / "logs" / "attempts.csv"
ATTEMPT_LOG_MIN_PERIOD_S = 2.0
ATTEMPT_LOG_DELTA_MM = 20.0
ATTEMPT_LOG_DELTA_DEG = 20.0
TARGET_MATCH_MAX_DIST_PX = 45.0
TARGET_HOLD_S = 0.60
TARGET_CENTER_EMA_ALPHA = 0.35
POSE_CONFIRMATION_FRAMES = 1
POSE_CONFIRM_DELTA_MM = 4.0
POSE_CONFIRM_DELTA_DEG = 6.0
TARGET_CONTROL_MIN_CONF = 0.60
TARGET_LOCK_CONF_TOLERANCE = 0.08
PICKED_TARGET_BLOCK_S = 2.5
PICKED_TARGET_BLOCK_RADIUS_PX = 80.0
CENTER_EMA_ALPHA = 0.08
U_EMA_ALPHA = 0.12
USE_EDGE_REFINEMENT = False

Z_MM_CONST = 53

DEFAULT_CALIBRATION_PATH = Path(__file__).resolve().parent / "calibracion" / "camera_calibration.yaml"
DEFAULT_FIXED_HOMOGRAPHY_PATH = Path(__file__).resolve().parent / "homography_table1.json"
HOMOGRAPHY_STATUS_ON = "H-FIJA: ON"
HOMOGRAPHY_STATUS_OFF = "H-FIJA: OFF"
HOMOGRAPHY_STATUS_INVALID = "H-FIJA: INVALIDA"
HOMOGRAPHY_STATUS_DELETED = "H-FIJA: borrada"

def set_hmi_coil(addr, value):
    with hmi_lock:
        hmi_coils[int(addr)] = 1 if int(value) else 0


def get_hmi_coil(addr):
    with hmi_lock:
        return int(hmi_coils.get(int(addr), 0))


def set_hmi_register(addr, value):
    with hmi_registers_lock:
        hmi_registers[int(addr)] = int(value) & 0xFFFF


def get_hmi_register(addr):
    with hmi_registers_lock:
        return int(hmi_registers.get(int(addr), 0))


def push_hmi_command(addr):
    with hmi_pending_lock:
        hmi_pending_cmds.add(int(addr))


def pop_hmi_command(addr):
    with hmi_pending_lock:
        addr = int(addr)
        if addr in hmi_pending_cmds:
            hmi_pending_cmds.remove(addr)
            return True
        return False


def set_hmi_speed(value):
    global speed_pending
    value = max(1, min(100, int(value)))
    set_hmi_register(REG_SPEED, value)
    with hmi_speed_lock:
        speed_pending = True


def pop_hmi_speed():
    global speed_pending
    with hmi_speed_lock:
        if speed_pending:
            speed_pending = False
            return get_hmi_register(REG_SPEED)
        return None


def send_coords_to_hmi(
    x_mm: float,
    y_mm: float,
    z_mm: float,
    u_deg: float,
):
    x = int(round(x_mm * 100)) & 0xFFFF
    y = int(round(y_mm * 100)) & 0xFFFF
    z = int(round(z_mm * 100)) & 0xFFFF
    u = int(round((u_deg % 360.0) * 100)) & 0xFFFF

    set_hmi_register(REG_X, x)
    set_hmi_register(REG_Y, y)
    set_hmi_register(REG_Z, z)
    set_hmi_register(REG_U, u)

    print(f"[HMI REG] X={x} Y={y} Z={z} U={u} disponibles para HMI")


def _recv_exact(conn: socket.socket, size: int) -> bytes:
    data = b""
    while len(data) < size:
        chunk = conn.recv(size - len(data))
        if not chunk:
            return b""
        data += chunk
    return data


def _build_modbus_response(transaction_id: int, protocol_id: int, unit_id: int, pdu: bytes) -> bytes:
    length = len(pdu) + 1
    mbap = struct.pack(">HHHB", transaction_id, protocol_id, length, unit_id)
    return mbap + pdu


def _handle_hmi_modbus_request(data: bytes) -> bytes:
    if len(data) < 8:
        return b""

    transaction_id, protocol_id, length, unit_id = struct.unpack(">HHHB", data[:7])
    pdu = data[7:]
    if len(pdu) < 1:
        return b""

    function_code = pdu[0]

    if function_code == 3:
        if len(data) < 12:
            return b""
        start_address = struct.unpack(">H", data[8:10])[0]
        quantity = struct.unpack(">H", data[10:12])[0]
        if HMI_DEBUG_READS:
            print(f"[HMI SERVER] Read Holding Registers start={start_address} quantity={quantity}")

        if quantity <= 0:
            err_pdu = struct.pack(">BB", function_code | 0x80, 3)
            return _build_modbus_response(transaction_id, protocol_id, unit_id, err_pdu)

        registers = []
        for i in range(quantity):
            addr = start_address + i
            val = get_hmi_register(addr)
            registers.append(val)
            if HMI_DEBUG_READS:
                print(f"[HMI SERVER] HR {addr} = {val}")

        byte_count = quantity * 2
        resp_pdu = struct.pack(">BB", function_code, byte_count)
        for val in registers:
            resp_pdu += struct.pack(">H", val)
        if HMI_DEBUG_READS:
            print("[HMI SERVER] Respuesta enviada func 03")
        return _build_modbus_response(transaction_id, protocol_id, unit_id, resp_pdu)

    if function_code == 5:
        if len(data) < 12:
            return b""
        coil_addr = struct.unpack(">H", data[8:10])[0]
        raw_value = struct.unpack(">H", data[10:12])[0]

        if raw_value == 0xFF00:
            coil_value = 1
        elif raw_value == 0x0000:
            coil_value = 0
        else:
            err_pdu = struct.pack(">BB", function_code | 0x80, 3)
            return _build_modbus_response(transaction_id, protocol_id, unit_id, err_pdu)

        set_hmi_coil(coil_addr, coil_value)

        btn_name = HMI_BUTTON_NAMES.get(coil_addr)
        if btn_name is not None:
            if coil_value == 1:
                push_hmi_command(coil_addr)
                print(f"[HMI CMD] {btn_name} recibido en ON")
            else:
                print(f"[HMI CMD] {btn_name} liberado")

        resp_pdu = pdu[:5]
        return _build_modbus_response(transaction_id, protocol_id, unit_id, resp_pdu)

    if function_code == 15:
        if len(pdu) < 6:
            return b""

        start_address = struct.unpack(">H", pdu[1:3])[0]
        quantity = struct.unpack(">H", pdu[3:5])[0]
        byte_count = pdu[5]
        expected_bytes = 6 + byte_count
        if len(pdu) < expected_bytes:
            return b""

        coil_bytes = pdu[6:6 + byte_count]
        for i in range(quantity):
            addr = start_address + i
            byte_idx = i // 8
            bit_idx = i % 8
            bit_val = (coil_bytes[byte_idx] >> bit_idx) & 0x01
            set_hmi_coil(addr, bit_val)

            btn_name = HMI_BUTTON_NAMES.get(addr)
            if btn_name is not None:
                if bit_val == 1:
                    push_hmi_command(addr)
                    print(f"[HMI CMD] {btn_name} recibido en ON")
                else:
                    print(f"[HMI CMD] {btn_name} liberado")

        resp_pdu = struct.pack(">BHH", function_code, start_address, quantity)
        return _build_modbus_response(transaction_id, protocol_id, unit_id, resp_pdu)

    if function_code == 6:
        if len(data) < 12:
            return b""
        register_addr = struct.unpack(">H", data[8:10])[0]
        register_value = struct.unpack(">H", data[10:12])[0]

        if register_addr == REG_SPEED:
            speed = max(1, min(100, int(register_value)))
            set_hmi_speed(speed)
            print(f"[HMI SPEED] Velocidad recibida desde HMI: {speed}")
        else:
            set_hmi_register(register_addr, register_value)

        resp_pdu = struct.pack(">BHH", function_code, register_addr, register_value)
        return _build_modbus_response(transaction_id, protocol_id, unit_id, resp_pdu)

    err_pdu = struct.pack(">BB", function_code | 0x80, 1)
    return _build_modbus_response(transaction_id, protocol_id, unit_id, err_pdu)


def start_hmi_modbus_server(host="0.0.0.0", port=HMI_SERVER_PORT):
    def _server_loop():
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            srv.bind((host, int(port)))
        except OSError as e:
            import errno
            if hasattr(e, 'errno') and e.errno == errno.EADDRINUSE:
                print(f"[HMI SERVER] Puerto {port} ya estÃƒÂ¡ en uso. Cierra la otra ejecuciÃƒÂ³n de binPicking.py o libera el puerto.")
                return
            else:
                raise
        srv.listen(5)
        print(f"[HMI SERVER] Escuchando en {host}:{port}")

        while True:
            conn, addr = srv.accept()
            try:
                while True:
                    header = _recv_exact(conn, 7)
                    if not header:
                        break

                    transaction_id, protocol_id, length, unit_id = struct.unpack(">HHHB", header)
                    if length <= 1:
                        break

                    pdu = _recv_exact(conn, length - 1)
                    if not pdu:
                        break

                    request = header + pdu
                    response = _handle_hmi_modbus_request(request)
                    if response:
                        conn.sendall(response)
            except Exception as exc:
                print(f"[HMI SERVER] Error con cliente {addr}: {exc}")
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

    thread = threading.Thread(target=_server_loop, daemon=True)
    thread.start()
    return thread
        
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="XY live + U live (stable + bias) + Z constante + words live")
    p.add_argument("--model", default=str(default_model_path()))
    p.add_argument("--conf", type=float, default=0.30)
    p.add_argument("--iou", type=float, default=0.40)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--zoom", type=float, default=DEFAULT_CAMERA_ZOOM)
    p.add_argument("--zoom-step", type=float, default=0.25)
    p.add_argument("--max-zoom", type=float, default=5.0)
    p.add_argument("--width", type=int, default=DEFAULT_CAMERA_WIDTH)
    p.add_argument("--height", type=int, default=DEFAULT_CAMERA_HEIGHT)
    p.add_argument("--fps", type=int, default=DEFAULT_CAMERA_FPS)
    p.add_argument("--camera-index", type=int, default=DEFAULT_CAMERA_INDEX)
    p.add_argument("--enable-autofocus", action="store_true")
    p.add_argument(
        "--calibration",
        default=str(DEFAULT_CALIBRATION_PATH),
        help="Archivo .yaml/.yml o .pkl con mtx y dist. Si no existe, no se aplica correccion.",
    )
    p.add_argument(
        "--undistort",
        action="store_true",
        help="Aplica correccion de distorsion antes de YOLO usando la calibracion cargada.",
    )
    p.add_argument("--host", default="192.168.250.10")
    p.add_argument("--port", type=int, default=502)
    p.add_argument("--unit-id", type=int, default=1)
    p.add_argument("--timeout", type=float, default=2.0)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--pulse", type=float, default=0.2)
    p.add_argument("--x-offset-mm", type=float, default=CAMERA_X_OFFSET_MM)
    p.add_argument("--y-offset-mm", type=float, default=CAMERA_Y_OFFSET_MM)
    p.add_argument("--send-period", type=float, default=WORDS_SEND_MIN_PERIOD_S)
    p.add_argument("--send-delta-mm", type=float, default=SEND_DELTA_MM)
    p.add_argument("--send-delta-deg", type=float, default=SEND_DELTA_DEG)
    p.add_argument("--confirm-frames", type=int, default=POSE_CONFIRMATION_FRAMES)
    p.add_argument("--disable-robot", action="store_true")
    return p.parse_args()


class EpsonClient(EpsonModbusClient):
    def pulse(self, addr: int, pulse_time: float = 0.15) -> None:
        self.write_bit(addr, 1)
        time.sleep(pulse_time)
        self.write_bit(addr, 0)

    def reset_all_outputs(self) -> None:
        for addr in [START, STOP, PAUSE, CONTINUE, RESET]:
            self.write_bit(addr, 0)

    @staticmethod
    def _to_uint16(value: int, signed: bool = False) -> int:
        value = int(value)
        if signed:
            if value < -32768 or value > 32767:
                raise ValueError(f"Valor signed fuera de rango en word: {value}")
            return value & 0xFFFF
        if value < 0 or value > 65535:
            raise ValueError(f"Valor fuera de rango en word: {value}")
        return value

    def write_register_int(self, addr: int, value: int, signed: bool = False) -> None:
        payload = self._to_uint16(value, signed=signed)
        last_exc = None

        if hasattr(self, "write_word"):
            try:
                self.write_word(addr, payload)
                return
            except Exception as exc:
                last_exc = exc

        if hasattr(self, "write_words"):
            try:
                self.write_words(addr, [payload])
                return
            except Exception as exc:
                last_exc = exc

        if hasattr(self, "write_register"):
            try:
                self.write_register(addr, payload)
                return
            except Exception as exc:
                last_exc = exc

        if last_exc is not None:
            raise last_exc
        raise AttributeError("EpsonModbusClient no tiene write_word/write_words/write_register")

    def write_pose_robot_mm_deg(self, x_mm: float, y_mm: float, z_mm: float, u_deg: float) -> None:
        z_send_mm = z_mm + Z_OFFSET_MM
        u_send_deg = normalize_deg_360(u_deg)

        x_i = int(round(float(x_mm) * POSE_SCALE))
        y_i = int(round(float(y_mm) * POSE_SCALE))
        z_i = int(round(float(z_send_mm) * POSE_SCALE))
        u_i = int(round(float(u_send_deg) * POSE_SCALE))

        self.write_register_int(REG_X, x_i, signed=True)
        self.write_register_int(REG_Y, y_i, signed=True)
        self.write_register_int(REG_Z, z_i, signed=True)
        self.write_register_int(REG_U, u_i, signed=False)


def clamp(v: float, vmin: float, vmax: float) -> float:
    return max(vmin, min(vmax, v))


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


def normalize_deg_180(angle: float) -> float:
    while angle > 180.0:
        angle -= 360.0
    while angle <= -180.0:
        angle += 360.0
    return angle


def normalize_deg_360(angle: float) -> float:
    return float(angle) % 360.0


def select_nearest_tab(instances_info, battery_center):
    best_tab = None
    best_dist2 = float("inf")

    for obj in instances_info:
        cls = _normalize_class_name(obj.get("class", ""))
        if cls not in TAB_CLASS_NAMES:
            continue

        tab_center = obj.get("center", None)
        if tab_center is None:
            continue

        dx = float(tab_center[0]) - float(battery_center[0])
        dy = float(tab_center[1]) - float(battery_center[1])
        dist2 = dx * dx + dy * dy

        if dist2 < best_dist2:
            best_dist2 = dist2
            best_tab = obj

    if best_tab is None:
        return None

    if best_dist2 > TAB_MATCH_MAX_DIST_PX * TAB_MATCH_MAX_DIST_PX:
        return None

    return best_tab


def tab_vector_to_robot_u_deg(battery_center, tab_center):
    dx = float(tab_center[0]) - float(battery_center[0])
    dy_img = float(tab_center[1]) - float(battery_center[1])

    # En imagen OpenCV, Y crece hacia abajo.
    # Para coordenadas tipo robot/mesa, Y debe crecer hacia arriba.
    dy_robot = -dy_img

    angle_deg = math.degrees(math.atan2(dy_robot, dx))
    return normalize_deg_360(angle_deg + TAB_U_BIAS_DEG)


def _normalize_class_name(name: str) -> str:
    s = str(name).strip().lower()
    s = s.replace("ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â±", "ÃƒÂ±")
    s = s.replace("ÃƒÆ’Ã‚Â±", "ÃƒÂ±")
    s = s.replace("ÃƒÂ±", "n")
    return s


def draw_clean_detections(
    img,
    result,
    mesa_idx: Optional[int],
    non_table_indices,
    selected_battery_idx: Optional[int] = None,
    selected_tab_idx: Optional[int] = None,
    overlap_battery_indices=None,
) -> any:
    out = img
    names = getattr(result, "names", {})
    font_scale = 0.42
    text_thickness = 1
    overlap_battery_indices = overlap_battery_indices or set()

    if mesa_idx is not None and mesa_idx >= 0 and mesa_idx < len(result.boxes):
        b = result.boxes[mesa_idx]
        xyxy = b.xyxy[0]
        x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy]
        conf = float(b.conf[0]) if b.conf is not None else 0.0
        cv2.rectangle(out, (x1, y1), (x2, y2), (160, 200, 255), 2)
        cv2.putText(
            out,
            f"mesa {conf:.2f}",
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (160, 200, 255),
            text_thickness,
            cv2.LINE_AA,
        )

    for idx in non_table_indices:
        if idx < 0 or idx >= len(result.boxes):
            continue

        b = result.boxes[idx]
        cls_id = int(b.cls[0]) if b.cls is not None else -1
        cls_name = _normalize_class_name(names.get(cls_id, str(cls_id)))
        if cls_name not in (BATTERY_CLASS_NAMES | TAB_CLASS_NAMES):
            continue

        xyxy = b.xyxy[0]
        x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy]
        conf = float(b.conf[0]) if b.conf is not None else 0.0

        if cls_name in BATTERY_CLASS_NAMES:
            color = (0, 255, 0) if idx == selected_battery_idx else (0, 220, 0)
            label = "bateria"
            box_thickness = 3 if idx == selected_battery_idx else 2
        else:
            color = (0, 128, 255) if idx == selected_tab_idx else (0, 150, 255)
            label = "pestana"
            box_thickness = 3 if idx == selected_tab_idx else 2

        cv2.rectangle(out, (x1, y1), (x2, y2), color, box_thickness)
        cv2.putText(
            out,
            f"{label} {conf:.2f}",
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            text_thickness,
            cv2.LINE_AA,
        )

    return out


def _battery_axis_vector_toward_tab(result, battery_obj, tab_center) -> Tuple[np.ndarray, Tuple[float, float]]:
    battery_center = battery_obj.get("center", None)
    if battery_center is None:
        raise ValueError("bateria sin centro")

    cx = float(battery_center[0])
    cy = float(battery_center[1])
    axis = np.asarray([1.0, 0.0], dtype=np.float64)
    has_axis = False

    det_idx = int(battery_obj.get("det_idx", -1))
    masks = getattr(result, "masks", None)
    if masks is not None and masks.xy is not None and 0 <= det_idx < len(masks.xy):
        pts = masks.xy[det_idx]
        if pts is not None and len(pts) >= 3:
            pts_arr = np.asarray(pts, dtype=np.float32)
            rect = cv2.minAreaRect(pts_arr.reshape(-1, 1, 2))
            box = cv2.boxPoints(rect).astype(np.float64)
            edges = [box[(i + 1) % 4] - box[i] for i in range(4)]
            lengths = [float(np.linalg.norm(e)) for e in edges]
            if lengths:
                long_edge = edges[int(np.argmax(lengths))]
                norm = float(np.linalg.norm(long_edge))
                if norm > 1e-9:
                    axis = long_edge / norm
                    has_axis = True

    if not has_axis:
        bbox = battery_obj.get("bbox", None)
        if bbox is None and 0 <= det_idx < len(result.boxes):
            xyxy = result.boxes[det_idx].xyxy[0]
            bbox = [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])]

        if bbox is not None:
            x1, y1, x2, y2 = [float(v) for v in bbox]
            w = max(1e-6, x2 - x1)
            h = max(1e-6, y2 - y1)
            axis = np.asarray([1.0, 0.0], dtype=np.float64) if w >= h else np.asarray([0.0, 1.0], dtype=np.float64)

    vtab = np.asarray([float(tab_center[0]) - cx, float(tab_center[1]) - cy], dtype=np.float64)
    if float(np.dot(axis, vtab)) < 0.0:
        axis = -axis

    norm_axis = float(np.linalg.norm(axis))
    if norm_axis <= 1e-9:
        axis = np.asarray([1.0, 0.0], dtype=np.float64)
    else:
        axis = axis / norm_axis

    return axis, (cx, cy)


def battery_axis_u_from_tab(result, battery_obj, tab_center):
    """
    Devuelve el angulo de la bateria usando su eje principal,
    orientado hacia el lado donde esta la pestana.
    No usa el vector centro bateria -> centro pestana como orientacion final.
    """
    axis, _ = _battery_axis_vector_toward_tab(result, battery_obj, tab_center)
    dx = float(axis[0])
    dy_robot = -float(axis[1])
    angle_axis_deg = math.degrees(math.atan2(dy_robot, dx))
    return normalize_deg_360(angle_axis_deg + BATTERY_AXIS_U_BIAS_DEG)


def point_inside_expanded_bbox(point, bbox, expand_px):
    px, py = float(point[0]), float(point[1])
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return (
        px >= x1 - expand_px and px <= x2 + expand_px and
        py >= y1 - expand_px and py <= y2 + expand_px
    )


def tab_inside_or_touching_battery_bbox(result, battery_obj, tab_obj):
    battery_bbox = get_obj_bbox_from_result(result, battery_obj)
    tab_center = tab_obj.get("center", None)

    if battery_bbox is None or tab_center is None:
        return False

    px, py = float(tab_center[0]), float(tab_center[1])
    x1, y1, x2, y2 = [float(v) for v in battery_bbox]

    expand = TAB_BATTERY_EXPAND_PX

    return (
        px >= x1 - expand and
        px <= x2 + expand and
        py >= y1 - expand and
        py <= y2 + expand
    )


def tab_center_inside_other_battery_bbox(result, instances_info, battery_obj, tab_obj):
    tab_center = tab_obj.get("center", None)
    if tab_center is None:
        return False

    px, py = float(tab_center[0]), float(tab_center[1])

    for other in instances_info:
        if other is battery_obj:
            continue

        cls = _normalize_class_name(other.get("class", ""))
        if cls not in BATTERY_CLASS_NAMES:
            continue

        other_bbox = get_obj_bbox_from_result(result, other)
        if other_bbox is None:
            continue

        x1, y1, x2, y2 = [float(v) for v in other_bbox]

        expand = TAB_BATTERY_EXPAND_PX
        if (
            px >= x1 - expand and px <= x2 + expand and
            py >= y1 - expand and py <= y2 + expand
        ):
            return True

    return False


def get_obj_bbox_from_result(result, obj):
    bbox = obj.get("bbox", None)
    if bbox is not None:
        return [float(v) for v in bbox]

    det_idx = int(obj.get("det_idx", -1))
    if det_idx >= 0 and det_idx < len(result.boxes):
        xyxy = result.boxes[det_idx].xyxy[0]
        return [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])]

    return None


def get_obj_mask_points(result, obj):
    det_idx = int(obj.get("det_idx", -1))
    masks = getattr(result, "masks", None)

    if masks is None or masks.xy is None:
        return None

    if det_idx < 0 or det_idx >= len(masks.xy):
        return None

    pts = masks.xy[det_idx]
    if pts is None or len(pts) < 3:
        return None

    return np.asarray(pts, dtype=np.float32)


def point_to_bbox_distance(point, bbox):
    px, py = float(point[0]), float(point[1])
    x1, y1, x2, y2 = [float(v) for v in bbox]

    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)

    return math.sqrt(dx * dx + dy * dy)


def point_to_mask_contour_distance(point, contour_pts):
    px, py = float(point[0]), float(point[1])
    contour = np.asarray(contour_pts, dtype=np.float32).reshape(-1, 1, 2)
    return abs(float(cv2.pointPolygonTest(contour, (px, py), True)))


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter
    if union <= 1e-6:
        return 0.0

    return inter / union


def battery_is_isolated(result, instances_info, battery_obj):
    bbox_a = get_obj_bbox_from_result(result, battery_obj)
    if bbox_a is None:
        return True

    for other in instances_info:
        if other is battery_obj:
            continue

        cls = _normalize_class_name(other.get("class", ""))
        if cls not in BATTERY_CLASS_NAMES:
            continue

        bbox_b = get_obj_bbox_from_result(result, other)
        if bbox_b is None:
            continue

        if bbox_iou(bbox_a, bbox_b) > PICK_OVERLAP_MAX_IOU:
            return False

    return True


def tab_is_closer_to_this_battery(result, instances_info, battery_obj, tab_obj):
    tab_center = tab_obj.get("center", None)
    if tab_center is None:
        return False

    this_mask = get_obj_mask_points(result, battery_obj)
    this_bbox = get_obj_bbox_from_result(result, battery_obj)

    if this_mask is not None:
        this_dist = point_to_mask_contour_distance(tab_center, this_mask)
    elif this_bbox is not None:
        this_dist = point_to_bbox_distance(tab_center, this_bbox)
    else:
        return False

    for other in instances_info:
        if other is battery_obj:
            continue

        cls = _normalize_class_name(other.get("class", ""))
        if cls not in BATTERY_CLASS_NAMES:
            continue

        other_mask = get_obj_mask_points(result, other)
        other_bbox = get_obj_bbox_from_result(result, other)

        if other_mask is not None:
            other_dist = point_to_mask_contour_distance(tab_center, other_mask)
        elif other_bbox is not None:
            other_dist = point_to_bbox_distance(tab_center, other_bbox)
        else:
            continue

        if other_dist + 3.0 < this_dist:
            return False

    return True


def validate_tab_belongs_to_battery(result, battery_obj, tab_obj):
    """
    Valida que la pestana pertenezca a esta bateria.

    Condiciones:
    1. La pestana debe estar cerca del contorno/mascara de la bateria.
    2. La pestana debe estar hacia uno de los extremos del eje largo de la bateria.
    3. La pestana no debe estar demasiado desviada lateralmente del eje de la bateria.
    """

    battery_center = battery_obj.get("center", None)
    tab_center = tab_obj.get("center", None)

    if battery_center is None or tab_center is None:
        return False

    battery_bbox = get_obj_bbox_from_result(result, battery_obj)
    if battery_bbox is None:
        return False

    try:
        axis, axis_center = _battery_axis_vector_toward_tab(result, battery_obj, tab_center)
    except Exception:
        return False

    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm <= 1e-6:
        return False
    axis = axis / norm

    perp = np.asarray([-axis[1], axis[0]], dtype=np.float64)

    bc = np.asarray([float(battery_center[0]), float(battery_center[1])], dtype=np.float64)
    tc = np.asarray([float(tab_center[0]), float(tab_center[1])], dtype=np.float64)
    v = tc - bc

    x1, y1, x2, y2 = [float(vv) for vv in battery_bbox]
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    long_len = max(w, h)
    short_len = min(w, h)

    proj = abs(float(np.dot(v, axis)))
    perp_dist = abs(float(np.dot(v, perp)))

    if proj < long_len * TAB_MIN_AXIS_PROJECTION_RATIO:
        return False

    if perp_dist > short_len * TAB_MAX_PERP_DIST_RATIO:
        return False

    battery_mask_pts = get_obj_mask_points(result, battery_obj)
    if battery_mask_pts is not None:
        edge_dist = point_to_mask_contour_distance(tab_center, battery_mask_pts)
    else:
        edge_dist = point_to_bbox_distance(tab_center, battery_bbox)

    if edge_dist > TAB_MAX_EDGE_DIST_PX:
        return False

    return True


def battery_inside_table_safe_zone(result, mesa_idx, battery_obj):
    if mesa_idx is None:
        return True

    battery_center = battery_obj.get("center", None)
    if battery_center is None:
        return False

    table_bbox = get_obj_bbox_from_result(result, {"det_idx": mesa_idx})
    if table_bbox is None:
        return True

    px, py = float(battery_center[0]), float(battery_center[1])
    x1, y1, x2, y2 = table_bbox

    return (
        px >= x1 + TABLE_EDGE_MARGIN_PX and
        px <= x2 - TABLE_EDGE_MARGIN_PX and
        py >= y1 + TABLE_EDGE_MARGIN_PX and
        py <= y2 - TABLE_EDGE_MARGIN_PX
    )


def get_candidate_tabs_for_battery(result, instances_info, battery_obj, reject_events=None):
    candidates = []

    battery_center = battery_obj.get("center", None)
    battery_bbox = get_obj_bbox_from_result(result, battery_obj)
    battery_mask_pts = get_obj_mask_points(result, battery_obj)

    if battery_center is None or battery_bbox is None:
        return candidates

    for obj in instances_info:
        cls = _normalize_class_name(obj.get("class", ""))
        if cls not in TAB_CLASS_NAMES:
            continue

        tab_conf = float(obj.get("conf", 0.0))
        if tab_conf < TAB_MIN_CONF:
            continue

        tab_center = obj.get("center", None)
        if tab_center is None:
            continue

        if not tab_inside_or_touching_battery_bbox(result, battery_obj, obj):
            if reject_events is not None:
                reject_events.append("pestana rechazada: fuera del bbox de su bateria")
            continue

        if tab_center_inside_other_battery_bbox(result, instances_info, battery_obj, obj):
            if reject_events is not None:
                reject_events.append("pestana rechazada: pertenece a otra bateria")
            continue

        dx = float(tab_center[0]) - float(battery_center[0])
        dy = float(tab_center[1]) - float(battery_center[1])
        center_dist = math.sqrt(dx * dx + dy * dy)

        if center_dist < TAB_CENTER_TO_BATTERY_CENTER_MIN_DIST_PX:
            continue

        if center_dist > TAB_MATCH_MAX_DIST_PX:
            continue

        valid_geom = validate_tab_belongs_to_battery(result, battery_obj, obj)
        closer_this = tab_is_closer_to_this_battery(result, instances_info, battery_obj, obj)

        if not closer_this:
            if reject_events is not None:
                reject_events.append("pestana rechazada: mas cerca de otra bateria")
            continue

        penalty = 0.0
        if not valid_geom:
            penalty += 80.0
            if reject_events is not None:
                reject_events.append("pestana candidata penalizada: edge/proj/perp")

        if battery_mask_pts is not None:
            edge_dist = point_to_mask_contour_distance(tab_center, battery_mask_pts)
            if edge_dist > TAB_TO_BATTERY_MASK_MAX_DIST_PX:
                if reject_events is not None:
                    reject_events.append(f"pestana rechazada: edge_dist={edge_dist:.2f}")
                continue
        else:
            edge_dist = point_to_bbox_distance(tab_center, battery_bbox)
            if edge_dist > TAB_BATTERY_EXPAND_PX:
                if reject_events is not None:
                    reject_events.append(f"pestana rechazada: bbox_dist={edge_dist:.2f}")
                continue

        try:
            axis, _ = _battery_axis_vector_toward_tab(result, battery_obj, tab_center)
            axis = np.asarray(axis, dtype=np.float64)
            axis = axis / max(np.linalg.norm(axis), 1e-6)
            perp = np.asarray([-axis[1], axis[0]], dtype=np.float64)
            bc = np.asarray([float(battery_center[0]), float(battery_center[1])], dtype=np.float64)
            tc = np.asarray([float(tab_center[0]), float(tab_center[1])], dtype=np.float64)
            v_axis = tc - bc
            proj = abs(float(np.dot(v_axis, axis)))
            perp_dist = abs(float(np.dot(v_axis, perp)))
        except Exception:
            proj = 0.0
            perp_dist = center_dist

        score = (
            perp_dist * 4.0
            - proj * 1.2
            + edge_dist * 2.0
            + center_dist * 0.03
            - tab_conf * 80.0
            + penalty
        )

        candidates.append((obj, score, edge_dist, center_dist, tab_conf))

    return candidates


def force_best_axis_tab_for_battery(result, instances_info, battery_obj):
    battery_center = battery_obj.get("center", None)
    if battery_center is None:
        return None

    battery_bbox = get_obj_bbox_from_result(result, battery_obj)
    if battery_bbox is None:
        return None

    try:
        dummy_tab = (float(battery_center[0]) + 100.0, float(battery_center[1]))
        axis, _ = _battery_axis_vector_toward_tab(result, battery_obj, dummy_tab)
    except Exception:
        x1, y1, x2, y2 = [float(v) for v in battery_bbox]
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        axis = np.asarray([1.0, 0.0], dtype=np.float64) if w >= h else np.asarray([0.0, 1.0], dtype=np.float64)

    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm <= 1e-6:
        return None
    axis = axis / norm

    perp = np.asarray([-axis[1], axis[0]], dtype=np.float64)

    bc = np.asarray([float(battery_center[0]), float(battery_center[1])], dtype=np.float64)

    x1, y1, x2, y2 = [float(v) for v in battery_bbox]
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    long_len = max(w, h)
    short_len = min(w, h)

    best_tab = None
    best_score = float("inf")

    for obj in instances_info:
        cls = _normalize_class_name(obj.get("class", ""))
        if cls not in TAB_CLASS_NAMES:
            continue

        tab_conf = float(obj.get("conf", 0.0))
        if tab_conf < TAB_MIN_CONF:
            continue

        tab_center = obj.get("center", None)
        if tab_center is None:
            continue

        if not tab_inside_or_touching_battery_bbox(result, battery_obj, obj):
            continue

        if tab_center_inside_other_battery_bbox(result, instances_info, battery_obj, obj):
            continue

        if not tab_is_closer_to_this_battery(result, instances_info, battery_obj, obj):
            continue

        tc = np.asarray([float(tab_center[0]), float(tab_center[1])], dtype=np.float64)
        v = tc - bc

        center_dist = float(np.linalg.norm(v))
        if center_dist > TAB_FORCE_MAX_DIST_PX:
            continue

        proj = abs(float(np.dot(v, axis)))
        perp_dist = abs(float(np.dot(v, perp)))

        end_penalty = 0.0
        if proj < long_len * 0.25:
            end_penalty += 80.0

        lateral_penalty = 0.0
        if perp_dist > short_len * 0.75:
            lateral_penalty += 100.0

        battery_mask_pts = get_obj_mask_points(result, battery_obj)
        if battery_mask_pts is not None:
            edge_dist = point_to_mask_contour_distance(tab_center, battery_mask_pts)
            if edge_dist > TAB_TO_BATTERY_MASK_MAX_DIST_PX:
                continue
        else:
            edge_dist = point_to_bbox_distance(tab_center, battery_bbox)
            if edge_dist > TAB_BATTERY_EXPAND_PX:
                continue

        score = (
            perp_dist * 4.0
            - proj * 1.2
            + edge_dist * 2.0
            + center_dist * 0.03
            - tab_conf * 80.0
            + end_penalty
            + lateral_penalty
        )

        if score < best_score:
            best_score = score
            best_tab = obj

    return best_tab


def match_tab_for_battery(result, instances_info, battery_obj, reject_events=None):
    candidates = get_candidate_tabs_for_battery(
        result,
        instances_info,
        battery_obj,
        reject_events=reject_events,
    )

    if not candidates:
        if FORCE_SELECT_NEAREST_TAB:
            tab = force_best_axis_tab_for_battery(result, instances_info, battery_obj)
            if tab is not None:
                if reject_events is not None:
                    reject_events.append("fallback: pestana por eje/extremo")
                tab_fallback = dict(tab)
                tab_fallback["__fallback__"] = True
                return tab_fallback
        return None

    best_tab, best_score, edge_dist, center_dist, tab_conf = min(
        candidates,
        key=lambda item: float(item[1])
    )

    return best_tab


def score_battery_tab_pair(result, battery, tab):
    battery_center = battery.get("center", None)
    tab_center = tab.get("center", None)

    if battery_center is None or tab_center is None:
        return float("inf")

    if not tab_inside_or_touching_battery_bbox(result, battery, tab):
        return float("inf")

    battery_conf = float(battery.get("conf", 0.0))
    tab_conf = float(tab.get("conf", 0.0))

    dx = float(tab_center[0]) - float(battery_center[0])
    dy = float(tab_center[1]) - float(battery_center[1])
    center_dist = math.sqrt(dx * dx + dy * dy)

    battery_bbox = get_obj_bbox_from_result(result, battery)
    battery_mask_pts = get_obj_mask_points(result, battery)
    if battery_mask_pts is not None:
        edge_dist = point_to_mask_contour_distance(tab_center, battery_mask_pts)
    else:
        edge_dist = point_to_bbox_distance(tab_center, battery_bbox) if battery_bbox is not None else 999.0

    return (
        edge_dist * 8.0
        + center_dist * 0.05
        - battery_conf * 30.0
        - tab_conf * 60.0
    )


def build_battery_tab_pairs(result, instances_info, mesa_idx, reject_events=None):
    pairs = []

    for battery in instances_info:
        cls = _normalize_class_name(battery.get("class", ""))
        if cls not in BATTERY_CLASS_NAMES:
            continue

        if float(battery.get("conf", 0.0)) < BATTERY_MIN_CONF:
            continue

        # Evita unir piezas cuando dos baterias aparecen fusionadas/solapadas.
        if not battery_is_isolated(result, instances_info, battery):
            if reject_events is not None:
                reject_events.append("bateria rechazada: muy cerca/solapada con otra")
            continue

        # Permisivo: no descartar por borde de mesa en este modo.

        tab = match_tab_for_battery(result, instances_info, battery, reject_events=reject_events)
        if tab is None:
            continue

        pair_score = score_battery_tab_pair(result, battery, tab)
        if not math.isfinite(pair_score):
            continue
        pairs.append((battery, tab, pair_score))

    return pairs


def select_stable_battery_tab_pair(pairs, last_center):
    if not pairs:
        return None, None

    if last_center is not None:
        best_pair = None
        best_dist2 = float("inf")

        for battery, tab, score in pairs:
            c = battery.get("center", None)
            if c is None:
                continue

            dx = float(c[0]) - float(last_center[0])
            dy = float(c[1]) - float(last_center[1])
            dist2 = dx * dx + dy * dy

            if dist2 < best_dist2:
                best_dist2 = dist2
                best_pair = (battery, tab)

        if best_pair is not None and best_dist2 <= TARGET_MATCH_MAX_DIST_PX * TARGET_MATCH_MAX_DIST_PX:
            return best_pair

    best = min(pairs, key=lambda pair: float(pair[2]))
    return best[0], best[1]


def circular_median_deg(samples: list[float]) -> float:
    if not samples:
        return 0.0
    ref = float(samples[0]) % 360.0
    deltas = [((float(angle) - ref + 180.0) % 360.0) - 180.0 for angle in samples]
    deltas.sort()
    median_delta = deltas[len(deltas) // 2]
    return (ref + median_delta) % 360.0


def stabilize_compass_deg(current: float, previous: Optional[float]) -> float:
    current = normalize_deg_360(current)
    if previous is None:
        return current
    alt = normalize_deg_360(current + 180.0)
    d_current = abs(((current - previous + 180.0) % 360.0) - 180.0)
    d_alt = abs(((alt - previous + 180.0) % 360.0) - 180.0)
    return current if d_current <= d_alt else alt


def ema_point(
    current: Tuple[float, float],
    previous: Optional[Tuple[float, float]],
    alpha: float,
) -> Tuple[float, float]:
    if previous is None:
        return float(current[0]), float(current[1])
    return (
        float(alpha) * float(current[0]) + (1.0 - float(alpha)) * float(previous[0]),
        float(alpha) * float(current[1]) + (1.0 - float(alpha)) * float(previous[1]),
    )


def ema_angle_deg(current: float, previous: Optional[float], alpha: float) -> float:
    current = normalize_deg_360(current)
    if previous is None:
        return current

    delta = ((current - previous + 180.0) % 360.0) - 180.0
    return normalize_deg_360(previous + float(alpha) * delta)


def angle_deg_to_word(angle_deg: float) -> int:
    angle_robot = normalize_deg_360(angle_deg)
    scaled = int(round(angle_robot * POSE_SCALE))
    if scaled < 0 or scaled > 65535:
        raise ValueError(f"Valor fuera de rango para word unsigned: {scaled}")
    return scaled


def compass_bearing_to_robot_u(bearing_deg: float) -> float:
    return normalize_deg_360(bearing_deg)


def compute_xy_robot_from_table_bbox_210mm_bottom_left(
    table_bbox_xyxy: Tuple[float, float, float, float],
    obj_center_px: Tuple[float, float],
) -> Tuple[float, float]:
    x1, y1, x2, y2 = table_bbox_xyxy
    px, py = float(obj_center_px[0]), float(obj_center_px[1])

    w = max(1.0, float(x2 - x1))
    h = max(1.0, float(y2 - y1))

    nx = clamp((px - x1) / w, 0.0, 1.0)
    ny = clamp((py - y1) / h, 0.0, 1.0)

    ny = 1.0 - ny
    return nx * TABLE_WIDTH_MM, ny * TABLE_HEIGHT_MM


def _default_table_dst_points() -> np.ndarray:
    return np.asarray(
        [
            [0.0, 0.0],
            [TABLE_WIDTH_MM, 0.0],
            [TABLE_WIDTH_MM, TABLE_HEIGHT_MM],
            [0.0, TABLE_HEIGHT_MM],
        ],
        dtype=np.float32,
    )


def _table_bbox_to_src_points(table_bbox_xyxy: Tuple[float, float, float, float]) -> np.ndarray:
    x1, y1, x2, y2 = [float(value) for value in table_bbox_xyxy]
    return np.asarray(
        [
            [x1, y2],
            [x2, y2],
            [x2, y1],
            [x1, y1],
        ],
        dtype=np.float32,
    )


def _is_valid_homography_matrix(homography: np.ndarray) -> bool:
    matrix = np.asarray(homography, dtype=np.float64)
    return matrix.shape == (3, 3) and np.isfinite(matrix).all()


def _is_valid_src_points(src_pts: np.ndarray) -> bool:
    points = np.asarray(src_pts, dtype=np.float64)
    return (
        points.ndim == 2
        and points.shape[1] == 2
        and points.shape[0] >= 4
        and np.isfinite(points).all()
    )


def _ordered_table_corners_bottom_left_origin(corners: np.ndarray) -> np.ndarray:
    pts = np.asarray(corners, np.float32).reshape(4, 2)

    y_sorted = pts[np.argsort(pts[:, 1])]
    top_two = y_sorted[:2]
    bottom_two = y_sorted[2:]

    top_left, top_right = sorted(top_two, key=lambda point: float(point[0]))
    bottom_left, bottom_right = sorted(bottom_two, key=lambda point: float(point[0]))

    return np.asarray([bottom_left, bottom_right, top_right, top_left], dtype=np.float32)


def compute_table_homography(result, mesa_idx: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    src_pts: Optional[np.ndarray] = None

    masks = getattr(result, "masks", None)
    if masks is not None and masks.xy is not None and mesa_idx < len(masks.xy):
        pts = masks.xy[mesa_idx]
        if pts is not None and len(pts) >= 3:
            pts_arr = np.asarray(pts, np.float32)
            rect = cv2.minAreaRect(pts_arr.reshape(-1, 1, 2))
            src_pts = _ordered_table_corners_bottom_left_origin(cv2.boxPoints(rect).astype(np.float32))

    if src_pts is None:
        box = result.boxes[mesa_idx].xyxy[0]
        src_pts = _table_bbox_to_src_points((float(box[0]), float(box[1]), float(box[2]), float(box[3])))

    if not _is_valid_src_points(src_pts):
        return None, None

    homography = cv2.getPerspectiveTransform(src_pts, _default_table_dst_points())
    if not _is_valid_homography_matrix(homography):
        return None, None

    return homography.astype(np.float32), src_pts.astype(np.float32)


def build_homography_metadata(
    zoom: float,
    width: int,
    height: int,
    undistort: bool,
) -> dict:
    return {
        "zoom": float(zoom),
        "width": int(width),
        "height": int(height),
        "undistort": bool(undistort),
    }


def save_fixed_homography_with_metadata(
    file_path: Path,
    homography: np.ndarray,
    src_pts: np.ndarray,
    metadata: dict,
    dst_pts: Optional[np.ndarray] = None,
) -> None:
    if not isinstance(metadata, dict):
        raise ValueError("metadata debe ser un diccionario")

    if not _is_valid_homography_matrix(homography):
        raise ValueError("La homografia debe ser una matriz 3x3 finita")
    if not _is_valid_src_points(src_pts):
        raise ValueError("src_pts debe contener al menos 4 puntos 2D finitos")

    payload = {
        "homography": np.asarray(homography, dtype=np.float64).tolist(),
        "src_pts": np.asarray(src_pts, dtype=np.float64).tolist(),
        "table_width_mm": TABLE_WIDTH_MM,
        "table_height_mm": TABLE_HEIGHT_MM,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "zoom": float(metadata.get("zoom")),
        "width": int(metadata.get("width")),
        "height": int(metadata.get("height")),
        "undistort": bool(metadata.get("undistort")),
    }

    if dst_pts is not None:
        payload["dst_pts_robot"] = np.asarray(dst_pts, dtype=np.float64).tolist()

    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"[INFO] Homografia guardada: {file_path}")


def load_fixed_homography(file_path: Path) -> Tuple[np.ndarray, np.ndarray, dict]:
    with file_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    homography = np.asarray(payload.get("homography"), dtype=np.float32)
    src_pts = np.asarray(payload.get("src_pts"), dtype=np.float32)
    metadata = {
        "zoom": payload.get("zoom"),
        "width": payload.get("width"),
        "height": payload.get("height"),
        "undistort": payload.get("undistort"),
    }

    if not _is_valid_homography_matrix(homography):
        raise ValueError("El archivo no contiene una homografia 3x3 valida")
    if not _is_valid_src_points(src_pts):
        raise ValueError("El archivo no contiene src_pts validos")
    if metadata["zoom"] is None or metadata["width"] is None or metadata["height"] is None or metadata["undistort"] is None:
        raise ValueError("El archivo no contiene metadata completa de configuracion")

    return homography, src_pts, metadata


def homography_metadata_matches(saved_metadata: dict, current_metadata: dict) -> bool:
    try:
        saved_zoom = float(saved_metadata.get("zoom"))
        current_zoom = float(current_metadata.get("zoom"))
        saved_width = int(saved_metadata.get("width"))
        current_width = int(current_metadata.get("width"))
        saved_height = int(saved_metadata.get("height"))
        current_height = int(current_metadata.get("height"))
        saved_undistort = bool(saved_metadata.get("undistort"))
        current_undistort = bool(current_metadata.get("undistort"))
    except (TypeError, ValueError):
        return False

    return (
        abs(saved_zoom - current_zoom) <= 1e-6
        and saved_width == current_width
        and saved_height == current_height
        and saved_undistort == current_undistort
    )


def delete_fixed_homography(file_path: Path) -> None:
    if file_path.exists():
        file_path.unlink()
    print("[INFO] Homografia fija borrada")


def compute_xy_robot_from_fixed_homography(
    homography: np.ndarray,
    obj_center_px: Tuple[float, float],
) -> Tuple[float, float]:
    if not _is_valid_homography_matrix(homography):
        raise ValueError("Homografia fija invalida")

    obj_pt = np.asarray([[[float(obj_center_px[0]), float(obj_center_px[1])]]], dtype=np.float32)
    mapped_pt = cv2.perspectiveTransform(obj_pt, np.asarray(homography, dtype=np.float32))
    if mapped_pt is None or not np.isfinite(mapped_pt).all():
        raise ValueError("La transformacion con homografia fija produjo valores invalidos")

    x_mm = clamp(float(mapped_pt[0, 0, 0]), 0.0, TABLE_WIDTH_MM)
    y_mm = clamp(float(mapped_pt[0, 0, 1]), 0.0, TABLE_HEIGHT_MM)
    return x_mm, y_mm


def compute_xy_robot_from_table_detection(
    result,
    mesa_idx: int,
    obj_center_px: Tuple[float, float],
) -> Tuple[float, float]:
    masks = getattr(result, "masks", None)
    if masks is None or masks.xy is None or mesa_idx >= len(masks.xy):
        box = result.boxes[mesa_idx].xyxy[0]
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        return compute_xy_robot_from_table_bbox_210mm_bottom_left((x1, y1, x2, y2), obj_center_px)

    pts = masks.xy[mesa_idx]
    if pts is None or len(pts) < 3:
        box = result.boxes[mesa_idx].xyxy[0]
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        return compute_xy_robot_from_table_bbox_210mm_bottom_left((x1, y1, x2, y2), obj_center_px)

    homography, _ = compute_table_homography(result, mesa_idx)
    if homography is None:
        box = result.boxes[mesa_idx].xyxy[0]
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        return compute_xy_robot_from_table_bbox_210mm_bottom_left((x1, y1, x2, y2), obj_center_px)

    return compute_xy_robot_from_fixed_homography(homography, obj_center_px)


def apply_camera_xy_offset(x_mm: float, y_mm: float, x_offset_mm: float, y_offset_mm: float) -> Tuple[float, float]:
    return (
        clamp(float(x_mm) + float(x_offset_mm), 0.0, TABLE_WIDTH_MM),
        clamp(float(y_mm) + float(y_offset_mm), 0.0, TABLE_HEIGHT_MM),
    )


def pose_changed_enough(a, b, delta_mm: float, delta_deg: float) -> bool:
    ax, ay, az, au = a
    bx, by, bz, bu = b
    return (
        abs(ax - bx) >= delta_mm
        or abs(ay - by) >= delta_mm
        or abs(az - bz) >= delta_mm
        or abs(au - bu) >= delta_deg
    )


def attempt_log_changed_enough(a, b) -> bool:
    ax, ay, az, au = a
    bx, by, bz, bu = b
    return (
        abs(ax - bx) >= ATTEMPT_LOG_DELTA_MM
        or abs(ay - by) >= ATTEMPT_LOG_DELTA_MM
        or abs(az - bz) >= ATTEMPT_LOG_DELTA_MM
        or abs(au - bu) >= ATTEMPT_LOG_DELTA_DEG
    )


def pose_close_for_confirmation(a, b) -> bool:
    ax, ay, az, au = a
    bx, by, bz, bu = b
    return (
        abs(ax - bx) <= POSE_CONFIRM_DELTA_MM
        and abs(ay - by) <= POSE_CONFIRM_DELTA_MM
        and abs(az - bz) <= POSE_CONFIRM_DELTA_MM
        and abs(au - bu) <= POSE_CONFIRM_DELTA_DEG
    )


def select_stable_battery_target(
    instances_info,
    last_center: Optional[Tuple[float, float]],
    blocked_center: Optional[Tuple[float, float]] = None,
    blocked_radius_px: float = 0.0,
) -> Optional[dict]:
    battery_candidates = [
        obj for obj in instances_info
        if _normalize_class_name(obj.get("class", "")) in BATTERY_CLASS_NAMES
    ]
    if blocked_center is not None and blocked_radius_px > 0.0:
        filtered_candidates = []
        blocked_radius2 = blocked_radius_px * blocked_radius_px
        for obj in battery_candidates:
            dx = float(obj["center"][0]) - float(blocked_center[0])
            dy = float(obj["center"][1]) - float(blocked_center[1])
            if (dx * dx + dy * dy) > blocked_radius2:
                filtered_candidates.append(obj)
        if filtered_candidates:
            battery_candidates = filtered_candidates

    if not battery_candidates:
        return None

    strong_candidates = [
        obj for obj in battery_candidates
        if float(obj.get("conf", 0.0)) >= TARGET_CONTROL_MIN_CONF
    ]
    if strong_candidates:
        battery_candidates = strong_candidates

    best_candidate = max(battery_candidates, key=lambda obj: float(obj.get("conf", 0.0)))
    if last_center is None:
        return best_candidate

    nearby_candidates = []
    for obj in battery_candidates:
        dx = float(obj["center"][0]) - float(last_center[0])
        dy = float(obj["center"][1]) - float(last_center[1])
        dist2 = dx * dx + dy * dy
        if dist2 <= (TARGET_MATCH_MAX_DIST_PX * TARGET_MATCH_MAX_DIST_PX):
            nearby_candidates.append((dist2, obj))

    if not nearby_candidates:
        return best_candidate

    nearby_best = max(nearby_candidates, key=lambda item: float(item[1].get("conf", 0.0)))[1]
    best_conf = float(best_candidate.get("conf", 0.0))
    nearby_conf = float(nearby_best.get("conf", 0.0))

    if nearby_conf >= (best_conf - TARGET_LOCK_CONF_TOLERANCE):
        return nearby_best
    return best_candidate


def send_robot_action(robot: Optional[EpsonClient], name: str, addr: int, pulse_time: float) -> str:
    if robot is None:
        return f"ROBOT: offline, {name} omitido"
    try:
        robot.reset_all_outputs()
        time.sleep(0.05)
        robot.pulse(addr, pulse_time=pulse_time)
        return f"ROBOT: {name} enviado"
    except Exception as exc:
        return f"ROBOT ERROR {name}: {exc}"


def append_attempt_log(
    status: str,
    reason: str,
    x_mm: Optional[float] = None,
    y_mm: Optional[float] = None,
    z_mm: Optional[float] = None,
    u_deg: Optional[float] = None,
) -> None:
    ATTEMPTS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not ATTEMPTS_LOG_PATH.exists():
        ATTEMPTS_LOG_PATH.write_text("timestamp,status,reason,x_mm,y_mm,z_mm,u_deg\n", encoding="utf-8")

    def _fmt(v: Optional[float], ndigits: int = 3) -> str:
        return "" if v is None else f"{float(v):.{ndigits}f}"

    ts = datetime.now().isoformat(timespec="seconds")
    row = (
        f"{ts},{status},{reason},"
        f"{_fmt(x_mm, 2)},{_fmt(y_mm, 2)},{_fmt(z_mm, 3)},{_fmt(u_deg, 2)}\n"
    )
    with ATTEMPTS_LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(row)


def draw_panel(img, zoom, u_send, robot_status, x_mm=None, y_mm=None, z_mm=None, u_deg=None):
    BLUE = (200, 90, 20)
    BLUE_DARK = (150, 55, 10)
    WHITE = (255, 255, 255)
    SOFT = (220, 230, 242)
    TEXT = (70, 70, 70)
    SUCCESS = (80, 170, 60)
    WARNING = (0, 170, 255)
    ERROR = (60, 60, 220)

    panel_width = 466
    panel = np.full((img.shape[0], panel_width, 3), 248, dtype=np.uint8)

    x, y = 18, 18
    w, h = panel_width - 36, 260
    header_h = 42

    overlay = panel.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), WHITE, -1)
    cv2.addWeighted(overlay, 0.90, panel, 0.10, 0, panel)

    cv2.rectangle(panel, (x, y), (x + w, y + h), SOFT, 1)

    cv2.rectangle(panel, (x, y), (x + w, y + header_h), BLUE, -1)
    cv2.putText(
        panel, "VISION + ROBOT", (x + 14, y + 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.70, WHITE, 2, cv2.LINE_AA
    )

    status_upper = robot_status.upper()
    if "ERROR" in status_upper:
        status_color = ERROR
    elif "OK" in status_upper or "CONECTADO" in status_upper or "ENVIADO" in status_upper:
        status_color = SUCCESS
    else:
        status_color = WARNING

    chip_x = x + w - 175
    chip_y = y + 9
    chip_w = 160
    chip_h = 24
    cv2.rectangle(panel, (chip_x, chip_y), (chip_x + chip_w, chip_y + chip_h), WHITE, -1)
    cv2.rectangle(panel, (chip_x, chip_y), (chip_x + chip_w, chip_y + chip_h), status_color, 1)

    chip_text = robot_status[:28]
    cv2.putText(
        panel, chip_text, (chip_x + 7, chip_y + 16),
        cv2.FONT_HERSHEY_SIMPLEX, 0.37, BLUE_DARK, 1, cv2.LINE_AA
    )

    content_x = x + 16
    row_y = y + 70

    def item(label, value, yy, accent=False):
        cv2.putText(
            panel, label, (content_x, yy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, BLUE_DARK, 1, cv2.LINE_AA
        )
        cv2.putText(
            panel, value, (content_x + 120, yy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.54,
            BLUE if accent else TEXT,
            2 if accent else 1,
            cv2.LINE_AA
        )

    item("Zoom", f"x{zoom:.2f}", row_y, accent=True)

    sep1 = row_y + 18
    cv2.line(panel, (x + 14, sep1), (x + w - 14, sep1), SOFT, 1, cv2.LINE_AA)

    coord_y = sep1 + 26
    cv2.putText(
        panel, "COORDENADAS", (content_x, coord_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, BLUE_DARK, 1, cv2.LINE_AA
    )

    if x_mm is not None and y_mm is not None and z_mm is not None and u_deg is not None:
        cv2.putText(
            panel,
            f"X: {x_mm:.1f}   Y: {y_mm:.1f}",
            (content_x, coord_y + 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.47, TEXT, 1, cv2.LINE_AA
        )
        u_text = f"Z: {z_mm:.3f}   U: {u_deg:.1f}"
        cv2.putText(
            panel,
            u_text,
            (content_x, coord_y + 48),
            cv2.FONT_HERSHEY_SIMPLEX, 0.47, TEXT, 1, cv2.LINE_AA
        )
    else:
        cv2.putText(
            panel,
            "Sin deteccion",
            (content_x, coord_y + 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.47, TEXT, 1, cv2.LINE_AA
        )

    sep2 = coord_y + 64
    cv2.line(panel, (x + 14, sep2), (x + w - 14, sep2), SOFT, 1, cv2.LINE_AA)

    cv2.putText(
        panel, "W/S: zoom",
        (content_x, sep2 + 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT, 1, cv2.LINE_AA
    )

    cv2.putText(
        panel, "1: START   2: STOP   3: PAUSE",
        (content_x, sep2 + 48),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT, 1, cv2.LINE_AA
    )

    cv2.putText(
        panel, "4: CONTINUE   5: RESET   H: guardar H   X: borrar H",
        (content_x, sep2 + 72),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT, 1, cv2.LINE_AA
    )

    cv2.line(panel, (0, 0), (0, panel.shape[0] - 1), (210, 210, 210), 2, cv2.LINE_AA)
    return cv2.hconcat([img, panel])


def main() -> int:
    cv2.namedWindow("XYZU LIVE (click aqui para teclado)")

    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] No se encontro el modelo: {model_path}")
        return 1

    model = YOLO(str(model_path))

    calibration_path = Path(args.calibration).expanduser()
    camera_matrix: Optional[np.ndarray] = None
    dist_coeffs: Optional[np.ndarray] = None
    undistort_status = "CALIB: desactivada"
    if calibration_path.exists():
        try:
            camera_matrix, dist_coeffs = load_camera_calibration(calibration_path)
            undistort_status = f"CALIB: cargada {calibration_path.name}"
        except Exception as exc:
            print(f"[WARN] No se pudo cargar calibracion {calibration_path}: {exc}")
            undistort_status = f"CALIB: error {type(exc).__name__}"
    else:
        undistort_status = f"CALIB: no existe {calibration_path.name}"

    camera, backend, _ = open_video_source(
        "", args.width, args.height, args.fps, args.camera_index, args.enable_autofocus
    )

    if camera is None:
        print("[ERROR] No se pudo abrir la camara.")
        return 1
    if not highgui_available():
        print("[ERROR] OpenCV sin soporte GUI.")
        return 1

    robot: Optional[EpsonClient] = None
    robot_status = "ROBOT: deshabilitado"
    if not args.disable_robot:
        robot = EpsonClient(
            host=args.host,
            port=args.port,
            unit_id=args.unit_id,
            timeout=args.timeout,
            offset=args.offset,
            pulse_time=args.pulse,
        )
        try:
            robot.connect()
            robot_status = f"ROBOT: conectado {args.host}:{args.port}"
        except Exception as exc:
            robot_status = f"ROBOT: no conecta -> {exc}"
            robot = None

    set_hmi_register(REG_SPEED, DEFAULT_SPEED)
    if robot is not None:
        try:
            robot.write_register_int(REG_SPEED, DEFAULT_SPEED, signed=False)
            print(f"[ROBOT SPEED] Velocidad inicial enviada al robot: {DEFAULT_SPEED}")
        except Exception as exc:
            print(f"[ROBOT SPEED ERROR] No se pudo inicializar velocidad: {exc}")

    start_hmi_modbus_server(host="0.0.0.0", port=HMI_SERVER_PORT)
    print("Servidor HMI listo. Configurar HMI hacia 192.168.250.2:1502")

    zoom = max(1.0, min(float(args.zoom), float(args.max_zoom)))
    zoom_step = max(0.05, float(args.zoom_step))
    max_zoom = max(1.0, float(args.max_zoom))
    current_homography_metadata = build_homography_metadata(
        zoom=zoom,
        width=args.width,
        height=args.height,
        undistort=bool(args.undistort),
    )
    fixed_homography_path = DEFAULT_FIXED_HOMOGRAPHY_PATH
    fixed_homography: Optional[np.ndarray] = None
    fixed_homography_src_pts: Optional[np.ndarray] = None
    homography_status = HOMOGRAPHY_STATUS_OFF
    homography_status_until = 0.0

    if fixed_homography_path.exists():
        try:
            fixed_homography_loaded, fixed_homography_src_pts_loaded, saved_homography_metadata = load_fixed_homography(fixed_homography_path)
            if homography_metadata_matches(saved_homography_metadata, current_homography_metadata):
                fixed_homography = fixed_homography_loaded
                fixed_homography_src_pts = fixed_homography_src_pts_loaded
                homography_status = HOMOGRAPHY_STATUS_ON
                print(f"[INFO] Homografia cargada: {fixed_homography_path}")
            else:
                fixed_homography = None
                fixed_homography_src_pts = None
                homography_status = HOMOGRAPHY_STATUS_INVALID
                print("[WARN] Homografia invalida por cambio de configuracion")
        except Exception as exc:
            homography_status = HOMOGRAPHY_STATUS_INVALID
            print(f"[WARN] No se pudo cargar la homografia fija {fixed_homography_path}: {exc}")

    last_sent_xyzu: Optional[Tuple[float, float, float, float]] = None
    last_sent_ts = 0.0
    robot_backoff_until = 0.0
    last_attempt_log_xyzu: Optional[Tuple[float, float, float, float]] = None
    last_attempt_log_ts = 0.0
    last_pair_debug_ts = 0.0

    last_compass: Optional[float] = None
    last_target_class: Optional[str] = None
    last_target_center: Optional[Tuple[float, float]] = None
    last_center_smoothed: Optional[Tuple[float, float]] = None
    last_u_smoothed: Optional[float] = None
    last_valid_xyzu: Optional[Tuple[float, float, float, float]] = None
    last_valid_detection_ts = 0.0
    orientation_window = 3
    min_orientation_samples = 1
    orientation_samples: list[float] = []
    blocked_target_center: Optional[Tuple[float, float]] = None
    blocked_target_until = 0.0

    u_deg_send = 0.0
    u_word_live = None

    x_mm_live = None
    y_mm_live = None

    no_battery_since = None
    stop_sent = False
    undistort_maps: Optional[Tuple[np.ndarray, np.ndarray]] = None
    undistort_map_size: Optional[Tuple[int, int]] = None

    try:
        while True:
            if backend == "picamera2":
                frame = camera.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                ok, frame = camera.read()
                if not ok:
                    break

            frame = apply_digital_zoom(frame, zoom)

            if args.undistort and camera_matrix is not None and dist_coeffs is not None:
                frame_size = (frame.shape[1], frame.shape[0])
                if undistort_maps is None or undistort_map_size != frame_size:
                    undistort_maps = build_undistort_maps(camera_matrix, dist_coeffs, frame_size)
                    undistort_map_size = frame_size
                map1, map2 = undistort_maps
                frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

            results = model.predict(frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz, verbose=False)
            result = results[0]

            mesa_idx = select_primary_table_index(result)
            non_table_indices = list_non_table_indices(result)

            instances_info = extract_instances_info(result, indices=non_table_indices)
            reject_events: list[str] = []
            pairs = build_battery_tab_pairs(result, instances_info, mesa_idx, reject_events=reject_events)
            target_obj, tab_obj = select_stable_battery_tab_pair(pairs, last_target_center)

            overlap_battery_indices = set()

            selected_battery_idx = int(target_obj.get("det_idx", -1)) if target_obj is not None else None
            selected_tab_idx = int(tab_obj.get("det_idx", -1)) if tab_obj is not None else None
            selected_pair_score = None
            used_fallback_pair = False
            if target_obj is not None and tab_obj is not None:
                t_bat_idx = int(target_obj.get("det_idx", -1))
                t_tab_idx = int(tab_obj.get("det_idx", -1))
                used_fallback_pair = bool(tab_obj.get("__fallback__", False))
                for battery, tab, pair_score in pairs:
                    b_idx = int(battery.get("det_idx", -1))
                    tb_idx = int(tab.get("det_idx", -1))
                    if b_idx == t_bat_idx and tb_idx == t_tab_idx:
                        selected_pair_score = float(pair_score)
                        break

            annotated = frame.copy()
            annotated = draw_clean_detections(
                annotated,
                result,
                mesa_idx,
                non_table_indices,
                selected_battery_idx=selected_battery_idx,
                selected_tab_idx=selected_tab_idx,
                overlap_battery_indices=overlap_battery_indices,
            )
            annotated = draw_table_workobject(annotated, result, mesa_idx, fixed_pose=None)

            if DEBUG_PAIR_LINES:
                for battery in instances_info:
                    cls = _normalize_class_name(battery.get("class", ""))
                    if cls not in BATTERY_CLASS_NAMES:
                        continue

                    c_bat = battery.get("center", None)
                    if c_bat is None:
                        continue

                    candidates = get_candidate_tabs_for_battery(result, instances_info, battery)
                    for tab, score, edge_dist, center_dist, tab_conf in candidates:
                        c_tab = tab.get("center", None)
                        if c_tab is None:
                            continue

                        cv2.line(
                            annotated,
                            (int(c_bat[0]), int(c_bat[1])),
                            (int(c_tab[0]), int(c_tab[1])),
                            (255, 180, 0),
                            1,
                            cv2.LINE_AA,
                        )

                        cv2.putText(
                            annotated,
                            f"{score:.1f}",
                            (int(c_tab[0]) + 4, int(c_tab[1]) + 4),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35,
                            (255, 180, 0),
                            1,
                            cv2.LINE_AA,
                        )

                if target_obj is not None and tab_obj is not None:
                    c_bat = target_obj.get("center", None)
                    c_tab = tab_obj.get("center", None)
                    if c_bat is not None and c_tab is not None:
                        cv2.line(
                            annotated,
                            (int(c_bat[0]), int(c_bat[1])),
                            (int(c_tab[0]), int(c_tab[1])),
                            (255, 0, 0),
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.circle(
                            annotated,
                            (int(c_tab[0]), int(c_tab[1])),
                            9,
                            (0, 140, 255),
                            3,
                            cv2.LINE_AA,
                        )
                        if selected_pair_score is not None:
                            cv2.putText(
                                annotated,
                                f"sel {selected_pair_score:.1f}",
                                (int(c_tab[0]) + 6, int(c_tab[1]) - 6),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.40,
                                (0, 140, 255),
                                1,
                                cv2.LINE_AA,
                            )

            now = time.time()
            battery_detected = target_obj is not None and tab_obj is not None
            hold_detection = False

            if PAIR_DEBUG_ENABLED and (now - last_pair_debug_ts) >= PAIR_DEBUG_PERIOD_S:
                num_bat = sum(1 for obj in instances_info if _normalize_class_name(obj.get("class", "")) in BATTERY_CLASS_NAMES)
                num_tab = sum(1 for obj in instances_info if _normalize_class_name(obj.get("class", "")) in TAB_CLASS_NAMES)
                sel_bat_center = target_obj.get("center", None) if target_obj is not None else None
                sel_tab_center = tab_obj.get("center", None) if tab_obj is not None else None
                print(
                    "[PAIR DEBUG] "
                    f"baterias={num_bat} pestanas={num_tab} parejas={len(pairs)} "
                    f"score={selected_pair_score if selected_pair_score is not None else 'NA'} "
                    f"bat_center={sel_bat_center} tab_center={sel_tab_center} "
                    f"rejects={reject_events[:3]}"
                )
                last_pair_debug_ts = now

            if not pairs:
                robot_status = "ROBOT: buscando pieza valida"
            elif target_obj is None or tab_obj is None:
                robot_status = "ROBOT: sin pieza valida"
            elif used_fallback_pair:
                robot_status = "ROBOT: pieza seleccionada por fallback"

            if battery_detected:
                no_battery_since = None
                stop_sent = False
            else:
                if no_battery_since is None:
                    no_battery_since = time.time()
                    print(f"Bateria no detectada, esperando {NO_BATTERY_STOP_DELAY_S:.0f} segundos antes de enviar STOP...")

                elapsed = time.time() - no_battery_since
                if elapsed >= NO_BATTERY_STOP_DELAY_S and not stop_sent:
                    print(f"No se detecto bateria durante {elapsed:.1f} segundos. Enviando STOP...")
                    robot_status = send_robot_action(robot, "STOP", STOP, args.pulse)
                    append_attempt_log(status="FAIL", reason="sin_deteccion_timeout")
                    stop_sent = True

                orientation_samples.clear()
                last_compass = None
                last_target_class = None
                last_target_center = None
                last_center_smoothed = None
                last_u_smoothed = None

            x_mm_live = None
            y_mm_live = None
            u_word_live = None

            if battery_detected and (fixed_homography is not None or mesa_idx is not None):
                if hold_detection and last_valid_xyzu is not None:
                    x_mm_live, y_mm_live, _, u_deg_send = last_valid_xyzu
                    u_word_live = angle_deg_to_word(u_deg_send)
                    robot_status = "ROBOT: target retenido"
                else:
                    obj0 = target_obj
                    if obj0 is None:
                        pass
                    elif _normalize_class_name(obj0.get("class", "")) in BATTERY_CLASS_NAMES:
                        curr_class = _normalize_class_name(obj0.get("class", "desconocido"))
                        curr_center = obj0.get("center", (0.0, 0.0))

                        curr_center = (float(curr_center[0]), float(curr_center[1]))

                        same_target = False
                        if last_target_class == curr_class and last_target_center is not None:
                            dx = float(curr_center[0]) - float(last_target_center[0])
                            dy = float(curr_center[1]) - float(last_target_center[1])
                            same_target = (dx * dx + dy * dy) <= (60.0 * 60.0)

                        if not same_target:
                            orientation_samples.clear()
                            last_compass = None
                            last_center_smoothed = None
                            last_u_smoothed = None

                        smooth_center = ema_point(
                            (float(curr_center[0]), float(curr_center[1])),
                            last_center_smoothed,
                            CENTER_EMA_ALPHA,
                        )
                        last_center_smoothed = smooth_center

                        tab_center = tab_obj.get("center", None) if tab_obj is not None else None

                        if tab_center is None:
                            robot_status = "ROBOT: pestana no detectada"
                            last_target_class = curr_class
                            last_target_center = smooth_center
                            stable_u = None
                        else:
                            raw_tab_u = battery_axis_u_from_tab(result, obj0, tab_center)

                            orientation_samples.append(raw_tab_u)
                            if len(orientation_samples) > orientation_window:
                                orientation_samples.pop(0)

                            measured_u_raw = circular_median_deg(orientation_samples)
                            u_smoothed = ema_angle_deg(measured_u_raw, last_u_smoothed, U_EMA_ALPHA)
                            last_u_smoothed = u_smoothed

                            stable_u = normalize_deg_360(u_smoothed)

                            last_compass = measured_u_raw
                            last_target_class = curr_class
                            last_target_center = smooth_center

                        c_bat = (int(smooth_center[0]), int(smooth_center[1]))
                        cv2.circle(annotated, c_bat, 7, (0, 255, 0), 2)

                        if tab_center is not None:
                            c_tab = (int(tab_center[0]), int(tab_center[1]))
                            u_diag_send = normalize_deg_360(stable_u) if stable_u is not None else None
                            axis_vec, axis_center = _battery_axis_vector_toward_tab(result, obj0, tab_center)
                            c_axis = (int(axis_center[0]), int(axis_center[1]))
                            axis_len = 70
                            c_axis_end = (
                                int(round(axis_center[0] + float(axis_vec[0]) * axis_len)),
                                int(round(axis_center[1] + float(axis_vec[1]) * axis_len)),
                            )
                            cv2.circle(annotated, c_tab, 7, (0, 128, 255), 2)
                            cv2.line(annotated, c_bat, c_tab, (180, 180, 80), 2)
                            cv2.arrowedLine(
                                annotated,
                                c_axis,
                                c_axis_end,
                                (0, 0, 255),
                                3,
                                tipLength=0.25,
                            )
                            cv2.putText(
                                annotated,
                                f"U={u_diag_send:.1f}" if u_diag_send is not None else "U=--",
                                (c_bat[0] + 10, c_bat[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (255, 255, 255),
                                2,
                                cv2.LINE_AA,
                            )
                        else:
                            cv2.putText(
                                annotated,
                                "SIN PESTANA",
                                (c_bat[0] + 10, c_bat[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 255),
                                2,
                                cv2.LINE_AA,
                            )

                        if fixed_homography is not None:
                            try:
                                x_mm, y_mm = compute_xy_robot_from_fixed_homography(
                                    fixed_homography,
                                    smooth_center,
                                )
                            except Exception as exc:
                                print(f"[WARN] Homografia fija invalida; se desactiva: {exc}")
                                fixed_homography = None
                                fixed_homography_src_pts = None
                                homography_status = HOMOGRAPHY_STATUS_OFF
                                if mesa_idx is not None:
                                    x_mm, y_mm = compute_xy_robot_from_table_detection(
                                        result,
                                        mesa_idx,
                                        smooth_center,
                                    )
                                else:
                                    x_mm, y_mm = None, None
                        else:
                            x_mm, y_mm = compute_xy_robot_from_table_detection(
                                result,
                                mesa_idx,
                                smooth_center,
                            )

                        if x_mm is None or y_mm is None:
                            pass
                        else:
                            x_mm, y_mm = apply_camera_xy_offset(
                                x_mm,
                                y_mm,
                                args.x_offset_mm,
                                args.y_offset_mm,
                            )

                            if last_valid_xyzu is not None:
                                dx = abs(float(x_mm) - float(last_valid_xyzu[0]))
                                dy = abs(float(y_mm) - float(last_valid_xyzu[1]))
                                if dx < 0.5 and dy < 0.5:
                                    x_mm = float(last_valid_xyzu[0])
                                    y_mm = float(last_valid_xyzu[1])

                            x_mm_live = x_mm
                            y_mm_live = y_mm

                            if stable_u is None:
                                x_mm_live = x_mm
                                y_mm_live = y_mm
                                robot_status = "ROBOT: bateria detectada, sin pestana"
                            else:
                                u_deg_send = normalize_deg_360(stable_u)
                                u_word_live = angle_deg_to_word(u_deg_send)
                                xyzu = (x_mm, y_mm, Z_MM_CONST, u_deg_send)

                                if len(orientation_samples) < min_orientation_samples:
                                    robot_status = (
                                        f"ROBOT: midiendo orientacion {len(orientation_samples)}/{orientation_window}"
                                    )
                                else:
                                    last_valid_xyzu = xyzu
                                    last_valid_detection_ts = now

                                if robot is not None and now >= robot_backoff_until and (now - last_sent_ts) >= args.send_period:
                                    should_send_xyzu = False
                                    if last_sent_xyzu is None:
                                        should_send_xyzu = True
                                    elif pose_changed_enough(last_sent_xyzu, xyzu, args.send_delta_mm, args.send_delta_deg):
                                        should_send_xyzu = True

                                    if should_send_xyzu:
                                        try:
                                            robot.write_pose_robot_mm_deg(*xyzu)
                                            # Enviar tambiÃƒÂ©n a la HMI
                                            send_coords_to_hmi(xyzu[0], xyzu[1], xyzu[2], xyzu[3])
                                            last_sent_xyzu = xyzu
                                            last_sent_ts = now
                                            robot_status = "ROBOT: words OK (live)"
                                            should_log_ok = False
                                            if last_attempt_log_xyzu is None:
                                                should_log_ok = True
                                            elif attempt_log_changed_enough(last_attempt_log_xyzu, xyzu):
                                                should_log_ok = True
                                            elif (now - last_attempt_log_ts) >= ATTEMPT_LOG_MIN_PERIOD_S:
                                                should_log_ok = True

                                            if should_log_ok:
                                                append_attempt_log(
                                                    status="OK",
                                                    reason="words_enviadas",
                                                    x_mm=x_mm,
                                                    y_mm=y_mm,
                                                    z_mm=Z_MM_CONST,
                                                    u_deg=u_deg_send,
                                                )
                                                last_attempt_log_xyzu = xyzu
                                                last_attempt_log_ts = now
                                        except Exception as exc:
                                            print(
                                                f"[ROBOT WORDS ERROR] X={x_mm:.2f} Y={y_mm:.2f} "
                                                f"Z={Z_MM_CONST:.3f} U_send={u_deg_send:.2f} WU={u_word_live} "
                                                f"-> {type(exc).__name__}: {exc}"
                                            )
                                            robot_status = f"ROBOT ERROR words: {type(exc).__name__}"
                                            append_attempt_log(
                                                status="FAIL",
                                                reason=f"words_error_{type(exc).__name__}",
                                                x_mm=x_mm,
                                                y_mm=y_mm,
                                                z_mm=Z_MM_CONST,
                                                u_deg=u_deg_send,
                                            )
                                            robot_backoff_until = now + 1.0

            annotated = draw_panel(
                annotated,
                zoom,
                u_deg_send,
                robot_status,
                x_mm_live,
                y_mm_live,
                Z_MM_CONST if x_mm_live is not None else None,
                u_deg_send if x_mm_live is not None else None,
            )

            if homography_status_until > 0.0 and now > homography_status_until:
                homography_status_until = 0.0
                if fixed_homography is not None:
                    homography_status = HOMOGRAPHY_STATUS_ON
                elif homography_status != HOMOGRAPHY_STATUS_INVALID:
                    homography_status = HOMOGRAPHY_STATUS_OFF

            if fixed_homography is not None and homography_status != HOMOGRAPHY_STATUS_DELETED:
                homography_status = HOMOGRAPHY_STATUS_ON
            elif fixed_homography is None and homography_status not in {HOMOGRAPHY_STATUS_INVALID, HOMOGRAPHY_STATUS_DELETED}:
                homography_status = HOMOGRAPHY_STATUS_OFF

            overlay_status = undistort_status if not args.undistort else f"{undistort_status} | UNDISTORT: ON"
            cv2.putText(
                annotated,
                f"{overlay_status} | {homography_status}",
                (14, annotated.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )

            if pop_hmi_command(HMI_START):
                print("[HMI CMD] Ejecutando START hacia robot")
                robot_status = send_robot_action(robot, "START", START, args.pulse)
                print(robot_status)

            if pop_hmi_command(HMI_STOP):
                print("[HMI CMD] Ejecutando STOP hacia robot")
                robot_status = send_robot_action(robot, "STOP", STOP, args.pulse)
                print(robot_status)

            if pop_hmi_command(HMI_PAUSE):
                print("[HMI CMD] Ejecutando PAUSE hacia robot")
                robot_status = send_robot_action(robot, "PAUSE", PAUSE, args.pulse)
                print(robot_status)

            if pop_hmi_command(HMI_CONTINUE):
                print("[HMI CMD] Ejecutando CONTINUE hacia robot")
                robot_status = send_robot_action(robot, "CONTINUE", CONTINUE, args.pulse)
                print(robot_status)

            if pop_hmi_command(HMI_RESET):
                print("[HMI CMD] Ejecutando RESET hacia robot")
                robot_status = send_robot_action(robot, "RESET", RESET, args.pulse)
                print(robot_status)

            speed_value = pop_hmi_speed()
            if speed_value is not None and robot is not None:
                try:
                    robot.write_register_int(REG_SPEED, speed_value, signed=False)
                    print(f"[ROBOT SPEED] Velocidad enviada al robot word 97: {speed_value}")
                except Exception as exc:
                    print(f"[ROBOT SPEED ERROR] No se pudo escribir velocidad {speed_value} en word 97: {exc}")

            cv2.imshow("XYZU LIVE (click aqui para teclado)", annotated)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q")):
                break

            if key in (ord("w"), ord("W")):
                zoom = min(max_zoom, zoom + zoom_step)
            elif key in (ord("s"), ord("S")):
                zoom = max(1.0, zoom - zoom_step)
            elif key == ord("0"):
                zoom = 1.0
            elif key in (ord("h"), ord("H")):
                if mesa_idx is None:
                    print("[WARN] No hay mesa detectada para capturar la homografia.")
                else:
                    homography, src_pts = compute_table_homography(result, mesa_idx)
                    if homography is None or src_pts is None:
                        print("[WARN] No se pudo calcular una homografia valida con la deteccion actual.")
                    else:
                        try:
                            save_fixed_homography_with_metadata(
                                fixed_homography_path,
                                homography,
                                src_pts,
                                current_homography_metadata,
                            )
                            fixed_homography = homography
                            fixed_homography_src_pts = src_pts
                            homography_status = HOMOGRAPHY_STATUS_ON
                            homography_status_until = 0.0
                            print(f"[INFO] Homografia fija activada: {fixed_homography_path.name}")
                        except Exception as exc:
                            print(f"[WARN] No se pudo guardar la homografia fija: {exc}")
            elif key in (ord("x"), ord("X")):
                try:
                    delete_fixed_homography(fixed_homography_path)
                    fixed_homography = None
                    fixed_homography_src_pts = None
                    homography_status = HOMOGRAPHY_STATUS_DELETED
                    homography_status_until = time.time() + 2.0
                except Exception as exc:
                    print(f"[WARN] No se pudo borrar la homografia fija: {exc}")
            elif key == ord("1"):
                if last_target_center is not None:
                    blocked_target_center = last_target_center
                    blocked_target_until = time.time() + PICKED_TARGET_BLOCK_S
                orientation_samples.clear()
                last_compass = None
                last_target_class = None
                last_target_center = None
                last_center_smoothed = None
                last_u_smoothed = None
                last_valid_xyzu = None
                robot_status = send_robot_action(robot, "START", START, args.pulse)
            elif key == ord("2"):
                robot_status = send_robot_action(robot, "STOP", STOP, args.pulse)
            elif key == ord("3"):
                robot_status = send_robot_action(robot, "PAUSE", PAUSE, args.pulse)
            elif key == ord("4"):
                robot_status = send_robot_action(robot, "CONTINUE", CONTINUE, args.pulse)
            elif key == ord("5"):
                robot_status = send_robot_action(robot, "RESET", RESET, args.pulse)

    finally:
        if backend == "picamera2":
            camera.stop()
        else:
            camera.release()
        cv2.destroyAllWindows()
        if robot is not None:
            try:
                robot.close()
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    sys.exit(main())