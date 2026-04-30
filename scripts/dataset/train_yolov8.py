# Entrenamiento YOLOv8 con el dataset descargado de Roboflow
from ultralytics import YOLO
import os


# Ruta al archivo data.yaml generado por Roboflow para el proyecto PruebaFinal-1
data_yaml = os.path.abspath(os.path.join(os.path.dirname(__file__), 'PruebaFinal-1/data.yaml'))

# Elige el modelo base (puedes cambiar a yolov8m/yolov8l/yolov8x si quieres)
model = YOLO('yolov8n-seg.pt')

# Entrena el modelo
model.train(
    data=data_yaml,
    epochs=10,
    imgsz=640,
    batch=8,
    project='runs/segment',
    name='roboflow_v1',
    exist_ok=True
)

# Puedes ajustar epochs, imgsz, batch según tu GPU y dataset.
