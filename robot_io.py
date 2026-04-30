#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from pymodbus.client import ModbusTcpClient


class EpsonModbusClient:
    def __init__(
        self,
        host="192.168.250.10",
        port=502,
        unit_id=1,
        timeout=2,
        offset=0,
        pulse_time=0.20,
    ):
        self.host = host
        self.port = port
        self.unit_id = unit_id
        self.timeout = timeout
        self.offset = offset
        self.pulse_time = pulse_time

        self.client = ModbusTcpClient(
            host=self.host,
            port=self.port,
            timeout=self.timeout,
        )

        # =========================
        # COILS (BITS)
        # visibles 520-524 => python -1
        # =========================
        self.START = 520 - 1      # 519
        self.STOP = 521 - 1       # 520
        self.PAUSE = 522 - 1      # 521
        self.CONTINUE = 523 - 1   # 522
        self.RESET = 524 - 1      # 523

        self.ALL_BITS = [self.START, self.STOP, self.PAUSE, self.CONTINUE, self.RESET]

        # =========================
        # HOLDING REGISTERS (WORDS)
        # visibles 32-35 => python -1
        # =========================
        self.REG_X = 32 - 1  # 31
        self.REG_Y = 33 - 1  # 32
        self.REG_Z = 34 - 1  # 33
        self.REG_U = 35 - 1  # 34

    def connect(self):
        ok = self.client.connect()
        if not ok:
            raise ConnectionError(f"No se pudo conectar a {self.host}:{self.port}")
        print(f"Conectado a {self.host}:{self.port}")
        return True

    def close(self):
        self.client.close()
        print("Conexion cerrada")

    def _addr(self, addr: int) -> int:
        return addr + self.offset

    def _call(self, method_name: str, **kwargs):
        method = getattr(self.client, method_name)
        try:
            rr = method(**kwargs, device_id=self.unit_id)
        except TypeError:
            rr = method(**kwargs, slave=self.unit_id)

        if rr is None:
            raise RuntimeError("Respuesta Modbus vacia")
        if hasattr(rr, "isError") and rr.isError():
            raise RuntimeError(f"Respuesta Modbus con error: {rr}")
        return rr

    # ---------- COILS ----------
    def write_bit(self, addr: int, value: bool):
        self._call(
            "write_coil",
            address=self._addr(addr),
            value=bool(value),
        )

    # ---------- WORDS (Holding Registers) ----------
    def write_word(self, addr: int, value: int):
        """
        Escribe 1 holding register (word).
        value debe ser 0..65535 (uint16)
        """
        value_i = int(value)
        if value_i < 0 or value_i > 65535:
            raise ValueError(f"Word fuera de rango: addr={addr}, value={value_i}")

        self._call(
            "write_register",
            address=self._addr(addr),
            value=value_i,
        )

    def write_words(self, addr: int, values: list[int]):
        """
        Escribe varios holding registers consecutivos.
        """
        vals = [int(v) for v in values]
        for v in vals:
            if v < 0 or v > 65535:
                raise ValueError(f"Word fuera de rango en write_words: {v}")

        self._call(
            "write_registers",
            address=self._addr(addr),
            values=vals,
        )

    # ---------- UTILIDADES ----------
    def reset_all(self):
        for addr in self.ALL_BITS:
            try:
                self.write_bit(addr, False)
            except Exception:
                pass
        time.sleep(0.08)

    def pulse_only(self, addr: int):
        self.reset_all()
        time.sleep(0.05)

        self.write_bit(addr, True)
        time.sleep(self.pulse_time)
        self.write_bit(addr, False)

        time.sleep(0.05)
        self.reset_all()

    def start(self):
        self.pulse_only(self.START)

    def stop(self):
        self.pulse_only(self.STOP)

    def pause(self):
        self.pulse_only(self.PAUSE)

    def continue_run(self):
        self.pulse_only(self.CONTINUE)

    def reset_robot(self):
        self.pulse_only(self.RESET)