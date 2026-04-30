#!/usr/bin/env python3
"""Descarga dataset de Roboflow en formato YOLOv8. SKvXxsCiwQjqMlw8cPtt"""

from __future__ import annotations

import importlib
import os
import sys
from getpass import getpass


def main() -> int:
	api_key = os.getenv("ROBOFLOW_API_KEY", "").strip()
	if not api_key:
		print("[INFO] No se encontro ROBOFLOW_API_KEY en variables de entorno.")
		try:
			api_key = getpass("Ingresa tu ROBOFLOW_API_KEY: ").strip()
		except (EOFError, KeyboardInterrupt):
			print("\n[ERROR] No se pudo leer la API key desde la terminal.")
			return 1
	if not api_key:
		print("[ERROR] API key vacia. No se puede continuar.")
		return 1

	try:
		module = importlib.import_module("roboflow")
		Roboflow = getattr(module, "Roboflow")
	except Exception as exc:
		print(f"[ERROR] No se pudo importar roboflow: {exc}")
		return 1

	try:
		rf = Roboflow(api_key=api_key)
		project = rf.workspace("josephs-workspace-gguso").project("pruebafinal-p1ezt")
		version = project.version(1)
		location = version.download("yolov8")
	except Exception as exc:
		print("[ERROR] Fallo la descarga del dataset desde Roboflow.")
		print(f"[ERROR] Detalle: {exc}")
		return 1

	print(f"[INFO] Dataset descargado correctamente en: {location}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())