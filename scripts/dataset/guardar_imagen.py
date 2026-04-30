from picamera2 import Picamera2
import cv2
import time
from pathlib import Path

# -----------------------------
# CONFIGURACIÓN
# -----------------------------
SAVE_DIR = Path("/home/human/BinPicking/Imagenes")
RES_W = 1280
RES_H = 720

ZOOM = 1.0
ZOOM_MIN = 1.0
ZOOM_MAX = 4.0
ZOOM_STEP = 0.2

SAVE_PREFIX = "dato"

SAVE_DIR.mkdir(parents=True, exist_ok=True)


def construir_vista_diagnostico(frame_rgb, clahe):
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    clahe_gray = clahe.apply(gray)
    edges = cv2.Canny(clahe_gray, 80, 160)

    h, w = gray.shape[:2]
    tile_w = max(320, w // 2)
    tile_h = max(180, h // 2)

    def to_bgr(img_gray):
        return cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    t1 = cv2.resize(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
    t2 = cv2.resize(to_bgr(gray), (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
    t3 = cv2.resize(to_bgr(clahe_gray), (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
    t4 = cv2.resize(to_bgr(edges), (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)

    cv2.putText(t1, "Original", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(t2, "Gray", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(t3, "CLAHE", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(t4, "Edges", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    diag = cv2.vconcat([cv2.hconcat([t1, t2]), cv2.hconcat([t3, t4])])

    mean_gray, std_gray = cv2.meanStdDev(gray)
    mean_clahe, std_clahe = cv2.meanStdDev(clahe_gray)
    edge_ratio = 100.0 * float(cv2.countNonZero(edges)) / float(edges.size)

    overlay = diag.copy()
    cv2.rectangle(overlay, (0, 0), (diag.shape[1], 48), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, diag, 0.55, 0, diag)

    metrics = (
        f"Gray mean={float(mean_gray[0][0]):.1f} std={float(std_gray[0][0]):.1f}   "
        f"CLAHE mean={float(mean_clahe[0][0]):.1f} std={float(std_clahe[0][0]):.1f}   "
        f"Edge%={edge_ratio:.2f}"
    )
    cv2.putText(diag, metrics, (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return diag

# -----------------------------
# FUNCIÓN DE ZOOM DIGITAL
# -----------------------------
def aplicar_zoom(frame, zoom_factor):
    h, w, _ = frame.shape

    new_w = int(w / zoom_factor)
    new_h = int(h / zoom_factor)

    x1 = (w - new_w) // 2
    y1 = (h - new_h) // 2
    x2 = x1 + new_w
    y2 = y1 + new_h

    cropped = frame[y1:y2, x1:x2]
    zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    return zoomed

# -----------------------------
# BUSCAR ÍNDICE INICIAL
# -----------------------------
i = 0
while (SAVE_DIR / f"{SAVE_PREFIX}_{i:04d}.jpg").exists():
    i += 1

# -----------------------------
# INICIALIZAR CÁMARA
# -----------------------------
print("Inicializando cámara...")
picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"size": (RES_W, RES_H), "format": "RGB888"}
)

picam2.configure(config)
picam2.start()

time.sleep(2)
print("Cámara iniciada.")
print("Controles:")
print("  g -> guardar foto")
print("  w -> zoom +")
print("  s -> zoom -")
print("  d -> diagnostico contraste")
print("  q -> salir")

show_diagnostics = False
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

try:
    while True:
        frame = picam2.capture_array()

        if frame is None:
            continue

        frame_zoom = aplicar_zoom(frame, ZOOM)
        vista = frame_zoom.copy()

        if show_diagnostics:
            diag_view = construir_vista_diagnostico(frame_zoom, clahe)
            cv2.imshow("DIAGNOSTICO CONTRASTE (D: on/off)", diag_view)

        # Texto en pantalla
        cv2.putText(
            vista,
            f"Zoom: {ZOOM:.1f}x",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        cv2.putText(
            vista,
            f"Guardadas: {i}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            vista,
            "g=guardar  w=zoom+  s=zoom-  d=diag  q=salir",
            (20, RES_H - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow("Captura dataset", cv2.cvtColor(vista, cv2.COLOR_RGB2BGR))

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("g"):
            filename = SAVE_DIR / f"{SAVE_PREFIX}_{i:04d}.jpg"
            ok = cv2.imwrite(str(filename), cv2.cvtColor(frame_zoom, cv2.COLOR_RGB2BGR))

            if ok:
                print(f"Guardada: {filename}")
                i += 1
            else:
                print("Error al guardar la imagen")

        elif key == ord("w"):
            ZOOM = min(ZOOM + ZOOM_STEP, ZOOM_MAX)
            print(f"Zoom: {ZOOM:.1f}x")

        elif key == ord("s"):
            ZOOM = max(ZOOM - ZOOM_STEP, ZOOM_MIN)
            print(f"Zoom: {ZOOM:.1f}x")

        elif key == ord("d"):
            show_diagnostics = not show_diagnostics
            if show_diagnostics:
                print("Diagnostico de contraste: ON")
            else:
                print("Diagnostico de contraste: OFF")
                try:
                    cv2.destroyWindow("DIAGNOSTICO CONTRASTE (D: on/off)")
                except cv2.error:
                    pass

except KeyboardInterrupt:
    print("\nPrograma detenido por el usuario.")

finally:
    try:
        picam2.stop()
    except Exception:
        pass

    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    print("Programa finalizado.")