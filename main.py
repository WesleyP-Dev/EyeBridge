"""EyeBridge — Windows e Raspberry Pi OS 64-bit. Execute --help."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
from collections import deque
import json
import logging
import math
from pathlib import Path
import signal
import sys
import time
from config import Settings
from core import LABELS, block_center
from display import screen_size, viewport_size
from timing import TimerResolution, FramePacer
from metrics import StageMetrics

ROOT = Path(__file__).resolve().parent
log = logging.getLogger("eyebridge")


def arguments():
    parser = argparse.ArgumentParser(description="EyeBridge · confirme mantendo o olhar por 5 segundos")
    parser.add_argument("--demo", action="store_true", help="olhar simulado, sem câmera")
    parser.add_argument('--mode', choices=('normal','lite','full'), default='normal', help=argparse.SUPPRESS)
    parser.add_argument("--demo-auto", action="store_true", help="percorre e confirma os oito itens")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--backend", choices=("auto", "opencv", "picamera2"), default="opencv")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--windowed", action="store_true", help="inicia em janela em vez de tela cheia")
    parser.add_argument("--headless", action="store_true", help="sem janela; requer --duration")
    parser.add_argument("--duration", type=float, default=0)
    parser.add_argument("--snapshot", type=Path, help="salva uma imagem somente ao encerrar")
    parser.add_argument("--report", type=Path, help="salva métricas e histórico em JSON local")
    parser.add_argument("--check", action="store_true", help="valida modelo sem abrir câmera")
    parser.add_argument("--diagnose-camera", action="store_true", help="lista formatos/controles no Linux, sem alterá-los")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration < 0 or (args.headless and args.duration <= 0):
        parser.error("Use --headless com --duration maior que zero")
    return args


def temperature():
    try:
        return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text())/1000
    except (OSError, ValueError):
        return None


def rate(timestamps):
    return (len(timestamps)-1)/(timestamps[-1]-timestamps[0]) if len(timestamps) > 1 and timestamps[-1] > timestamps[0] else 0.0


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    args = arguments()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        settings = Settings.load(args.config)
        args.calibration = args.calibration or ROOT/'calibration-normal.json'
    except (OSError, ValueError, TypeError) as exc:
        log.error("Configuração inválida: %s", exc)
        return 2
    if args.diagnose_camera:
        from camera_diagnostics import inspect_camera
        report = inspect_camera(args.camera)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if args.report:
            write_json(args.report, report)
        return 0
    try:
        import cv2
        from camera_stream import CameraCaptureWorker
        from detector import EyeDetector
        from hud import HUD
        from vision_worker import VisionWorker
        from interaction import InteractionSession
    except ImportError as exc:
        log.error("Dependência ausente: %s. Execute o instalador.", exc)
        return 2
    cv2.setNumThreads(settings.cv_threads)
    cv2.ocl.setUseOpenCL(False)
    if args.check:
        try:
            EyeDetector.preflight()
            print(json.dumps({"python": sys.version.split()[0], "opencv": cv2.__version__,
                              "model": "Face Landmarker + sinais oculares: OK"}, ensure_ascii=False))
            return 0
        except Exception as exc:
            log.error("Falha ao carregar modelo: %s", exc, exc_info=args.verbose)
            return 2
    if not (args.demo or args.demo_auto):
        try:
            log.info("Validando MediaPipe e modelo antes de abrir a câmera...")
            EyeDetector.preflight()
        except Exception as exc:
            log.error("Detector não inicializado: %s", exc, exc_info=True)
            if args.report:
                write_json(args.report, {"exit_code": 2, "detector_error": str(exc),
                                         "camera_opened": False, "processing_fps_observed": 0})
            return 2
    session = InteractionSession(settings, args.calibration, args.demo or args.demo_auto)
    hud, ui_metrics = HUD(max_pixels=settings.ui_max_pixels), StageMetrics()
    hud.mode = 'normal'
    camera = vision = None
    stopped = False
    def stop(*_):
        nonlocal stopped
        stopped = True
    previous_signals = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    timer = TimerResolution().start()
    pacer = FramePacer(60)
    window = "EyeBridge"
    mirrored, fullscreen, preview = True, settings.fullscreen and not args.windowed, settings.preview_enabled
    last_frame, canvas = None, None
    next_render = next_window_check = temp_at = 0.0
    last_draw_key = None
    cpu_temp = None
    display_times = deque(maxlen=90)
    start = time.perf_counter()
    exit_code = 0
    final_vision_fps = 0.0
    view = session.snapshot()
    def render(now, final=False):
        return hud.render(last_frame.image if last_frame is not None and preview else None,
                          view.engine, camera.status, vision.status, demo=camera.demo,
                          mirrored=mirrored, fps=final_vision_fps if final else vision.fps, latency=vision.latency_ms,
                          temperature=cpu_temp,
                          calibration=None if vision.error_message else view.calibration,
                          message="ERRO NO DETECTOR · veja o terminal / execute --check" if vision.error_message else view.message,
                          camera_fps=camera.fps, now=now, preview_visible=preview,
                          display_fps=rate(display_times))
    try:
        camera = CameraCaptureWorker(settings, args.camera, args.backend, session.demo).start()
        vision = VisionWorker(camera, settings, args.demo_auto, session=session).start()
        if not args.headless:

            cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
            if fullscreen:
                hud.resize(*screen_size())
                cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.resizeWindow(window, hud.width, hud.height)
            def mouse(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN and hud.clear_hit(x, y):
                    session.clear()
                elif camera.demo and event == cv2.EVENT_MOUSEMOVE:
                    vision.move_demo(*hud.demo_position(x, y))
            cv2.setMouseCallback(window, mouse)
        while not stopped:
            now = time.perf_counter()
            if args.duration and now-start >= args.duration:
                break
            if now-temp_at >= 2:
                cpu_temp, temp_at = temperature(), now
            last_frame = camera.latest()
            if not args.headless:
                snapshot_start = time.perf_counter()
                view = session.snapshot()
                ui_metrics.record(snapshot_ms=(time.perf_counter()-snapshot_start)*1000)
                draw_key = (last_frame.sequence if last_frame else -1, view.revision, int(now*2), mirrored, preview)
                if now >= next_render and draw_key != last_draw_key:
                    render_start = time.perf_counter()
                    canvas = render(now)
                    ui_metrics.record(hud_ms=(time.perf_counter()-render_start)*1000)
                    present_start = time.perf_counter()

                    cv2.imshow(window, canvas)
                    ui_metrics.record(present_call_ms=(time.perf_counter()-present_start)*1000)
                    display_times.append(time.perf_counter())
                    last_draw_key = draw_key
                    next_render = max(next_render+1/settings.display_fps, now+0.001)
                event_start = time.perf_counter()
                key = cv2.waitKeyEx(1)
                ui_metrics.record(event_pump_ms=(time.perf_counter()-event_start)*1000)
                if now >= next_window_check:
                    try:
                        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                            break
                    except cv2.error:
                        break
                    viewport = viewport_size(cv2, window)
                    if viewport and hud.resize(*viewport):
                        last_draw_key = None
                        log.info("Interface adaptada a %sx%s; canvas %sx%s", *viewport, hud.width, hud.height)
                    next_window_check = now+0.3
                if key in (27, ord("q"), ord("Q")):
                    break
                if key >= 0:
                    char = chr(key & 0xff).lower()
                    if char == "d":
                        if camera.demo:
                            try:
                                EyeDetector.preflight()
                            except Exception as exc:
                                log.error("Não foi possível ativar câmera real: %s", exc, exc_info=True)
                                session.detector_failed("MediaPipe indisponível. Reinicie e execute --check.")
                                continue
                        session.set_demo(not camera.demo)
                        camera.toggle_demo()
                        last_frame = None
                    elif char in ("c", "g"):
                        if vision.error_message and not camera.demo:
                            session.detector_failed("Detector indisponível. Reinicie e execute --check antes de calibrar.")
                        else:
                            session.calibrate(char == "g")
                    elif char in ("+", "=", "-"):
                        session.gain(char != "-")
                    elif char == "f":
                        fullscreen = not fullscreen
                        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN,
                                              cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
                    elif char == "m":
                        mirrored = not mirrored
                    elif char == "v":
                        preview = not preview
                    elif char == "h":
                        session.clear()
                    elif camera.demo:
                        if char in "1234":
                            vision.move_demo(.2+.2*(int(char)-1),.5)
            time.sleep(max(0.001, pacer.delay()))
    except (RuntimeError, cv2.error, OSError) as exc:
        log.exception("Falha na execução: %s", exc)
        exit_code = 1
    finally:


        final_vision_fps = vision.fps if vision else 0.0
        if vision:
            vision.close()
        if camera:
            camera.close()
        timer.close()
        if not args.headless:
            cv2.destroyAllWindows()
        for sig, handler in previous_signals.items():
            signal.signal(sig, handler)
    view = session.snapshot()
    if args.snapshot and camera and vision:
        canvas = render(time.perf_counter(), final=True)
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        ok, encoded = cv2.imencode(".png", canvas)
        if ok:
            encoded.tofile(str(args.snapshot))
    if args.report:
        vision_stages = vision.metrics.summary() if vision else {}
        def median(name):
            return vision_stages.get(name, {}).get("median")
        hints = []
        capture_fps = camera.fps if camera else 0
        if 0 < capture_fps < settings.camera_fps*0.75:
            hints.append("Captura abaixo da meta: compare formatos, exposição e iluminação com --diagnose-camera.")
        if camera and camera.scene_brightness is not None and camera.scene_brightness < 70:
            hints.append("Cena com luminosidade média baixa; ilumine o rosto antes de reduzir a exposição.")
        if vision and capture_fps and final_vision_fps < capture_fps*0.8:
            hints.append("Visão abaixo da taxa capturada: verifique model_ms e eye_geometry_ms.")
        report = {"app_version":"3.10","duration_s": round(time.perf_counter()-start, 2), "demo": bool(camera and camera.demo),
                  "processing_fps_observed": round(final_vision_fps, 2),
                  "camera_fps_observed": round(capture_fps, 2), "camera_fps_requested": settings.camera_fps,
                  "processing_fps_target": settings.processing_fps, "display_fps_target": settings.display_fps,
                  "confirmation_seconds": settings.confirmation_seconds,
                  "canvas_size": [hud.width, hud.height], "fullscreen": fullscreen,
                  "ui_max_pixels": settings.ui_max_pixels,
                  "mode": "normal",
                  "tracking_quality_message": view.engine.quality_message,
                  "display_submissions_fps": round(rate(display_times), 2),
                  "inference_median_ms": median("inference_ms"),
                  "inference_p95_ms": vision_stages.get("inference_ms", {}).get("p95"),
                  "temperature_c": cpu_temp, "camera_mode": camera.mode if camera else {},
                  "scene_brightness_0_255": camera.scene_brightness if camera else None,
                  "frames_processed": vision.processed_frames if vision else 0,
                  "frames_with_face": vision.face_frames if vision else 0,
                  "frames_with_consistent_gaze": vision.gaze_frames if vision else 0,
                  "stages": {"capture": camera.metrics.summary() if camera else {},
                             "vision": vision_stages, "interface": ui_metrics.summary()},
                  "bottleneck_hints": hints, "region_changes": view.region_changes,
                  "regions_seen": view.regions_seen,
                  "selected_region": LABELS[view.engine.menu.selected] if view.engine.menu.selected is not None else None,
                  "confirmation_history": [{"item": e.label, "time": e.clock} for e in view.engine.history.entries],
                  "detector_error": vision.error_message if vision else None,
                  "detector": vision.status if vision else "indisponível", "exit_code": exit_code}
        write_json(args.report, report)
        log.info("Relatório salvo: %s", args.report)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
