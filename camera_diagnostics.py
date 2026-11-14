"""Consulta os dispositivos e formatos da câmera."""
import os
from pathlib import Path
import platform
import shutil
import subprocess


def inspect_camera(index):
    report = {"system": platform.system(), "architecture": platform.machine(),
              "camera_index": index, "device": f"/dev/video{index}",
              "display_session": os.environ.get("XDG_SESSION_TYPE", "não informado"),
              "wayland": bool(os.environ.get("WAYLAND_DISPLAY")), "commands": {}}
    if platform.system() != "Linux":
        report["note"] = "V4L2 é específico do Linux; use o relatório de execução para a webcam no Windows."
        return report
    name = Path(f"/sys/class/video4linux/video{index}/name")
    if name.exists():
        report["device_name"] = name.read_text(errors="replace").strip()
    executable = shutil.which("v4l2-ctl")
    if not executable:
        report["note"] = "Instale v4l-utils: sudo apt install v4l-utils"
        return report
    queries = {"devices": ["--list-devices"],
               "configuration": ["-d", report["device"], "--all"],
               "formats": ["-d", report["device"], "--list-formats-ext"],
               "controls": ["-d", report["device"], "--list-ctrls"]}
    for key, arguments in queries.items():
        try:
            result = subprocess.run([executable, *arguments], capture_output=True, text=True,
                                    encoding="utf-8", errors="replace", timeout=5)
            report["commands"][key] = {"returncode": result.returncode,
                                       "output": (result.stdout+result.stderr)[:14000]}
        except (OSError, subprocess.TimeoutExpired) as exc:
            report["commands"][key] = {"error": str(exc)}
    report["note"] = ("Consultas somente de leitura. Exposure auto priority pode permitir redução de FPS; "
                      "não foi modificada para evitar escurecer a imagem dos olhos.")
    return report
