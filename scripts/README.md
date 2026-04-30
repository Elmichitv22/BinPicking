# Scripts

Estructura pensada para separar herramientas por uso.

## calibration/
- `capturar_imagenes_calibracion.py`: captura imagenes del tablero
- `calibrar_desde_imagenes.py`: calcula la calibracion intrinseca y guarda un reporte de imagenes validas
- `previsualizar_undistort.py`: compara original vs corregida usando por defecto las imagenes validas del ultimo calibrado; si no hay reporte, usa `calibracion/calibracion_imgs`

## dataset/
- `guardar_imagen.py`: captura imagenes para dataset manual
- `dataset_dl.py`: descarga dataset desde Roboflow

## Ejecucion
- `python -m scripts.calibration.capturar_imagenes_calibracion`
- `python -m scripts.calibration.calibrar_desde_imagenes`
- `python -m scripts.calibration.previsualizar_undistort`
- `python -m scripts.calibration.previsualizar_undistort --live`
- `python -m scripts.dataset.guardar_imagen`
- `python -m scripts.dataset.dataset_dl`
