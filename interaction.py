"""Interação executada na thread de visão; a GUI usa apenas cópias de estado."""
from copy import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import threading
import time
import json
from core import Calibration, Calibrator, InteractionEngine, Observation
from horizontal_calibration import HorizontalCalibrator


class InteractionSession:
    def __init__(self, settings, calibration_path, demo=False, clock=None):
        self.s = settings
        self.clock = clock or time.perf_counter
        self.path = Path(calibration_path)
        self.lock = threading.RLock()
        self.demo = demo
        self.real_profile = Calibration(ear_open=settings.ear_open)
        self.message = "Pressione G para calibrar o olhar horizontal."
        if self.path.exists():
            try:
                self.real_profile = Calibration.load(self.path)
                self.message = "Calibração carregada. G para refazer." if self.real_profile.mapping else "Perfil antigo: G para nova calibração 2D."
            except (OSError, ValueError, TypeError):
                self.message = "Calibração inválida. Pressione G para refazer."
        self.engine = InteractionEngine(settings, (Calibration.paged_demo() if settings.paged_navigation else Calibration.demo()) if demo else self.real_profile)
        self.engine.paged_ui=settings.paged_navigation
        self.calibrator = None
        self.revision = 0
        self.region_changes = 0
        self.regions_seen = set()
        self.last_observation = None
        if demo:
            self.message = "DEMO: mouse ou 1–4; olhe para as bordas para trocar de página."

    def consume(self, frame, observation):
        now = self.clock()
        with self.lock:
            if frame.demo != self.demo:
                return
            self.last_observation = observation.timestamp
            obs = observation if now-observation.timestamp <= self.s.max_gap else Observation(now)
            old_confirmation = self.engine.last_confirmation
            if self.calibrator:
                self.engine.update(obs, paused=True)
                try:
                    result = self.calibrator.update(obs, self.engine.calibration)
                    if result:
                        self._save_diagnostic()
                        result.save(self.path)
                        self.real_profile = result
                        self.engine.calibration = result
                        self.engine.reset()
                        self.calibrator = None
                        self.message = "Calibração concluída e salva."
                except (ValueError, OSError) as exc:
                    self._save_diagnostic(str(exc))
                    self.calibrator = None
                    self.message = f"Calibração não salva: {exc}"
            else:
                previous = self.engine.menu.selected
                mapping = self.engine.calibration.mapping
                horizontal=bool(mapping and mapping.get('horizontal'))
                incompatible=mapping is None or (not horizontal and bool(obs.features) and len(mapping['mean'])!=len(obs.features))
                if len(obs.features)==6 and self.s.paged_navigation and not horizontal:incompatible=True
                if len(obs.features)==6 and not self.s.paged_navigation:
                    incompatible=not (mapping and mapping.get('classifier') and mapping['classifier'].get('validation_protocol')=='independent_revisit')
                if not self.demo and incompatible and obs.valid:
                    self.message = "Pressione G: nova calibração do rastreador necessária."
                selected=self.engine.update(obs,paused=not self.demo and incompatible)
                if selected is not None:
                    self.regions_seen.add(selected+1)
                    if selected != previous:
                        self.region_changes += 1
                if old_confirmation is not self.engine.last_confirmation:
                    self.message = ""
            self.revision += 1

    def _save_diagnostic(self, error=None):
        """Somente medidas numéricas; não salva imagens nem substitui perfil válido."""
        if self.calibrator is None:
            return
        report = dict(self.calibrator.diagnostic)
        report['error'] = error
        try:
            path = self.path.with_name(self.path.stem+'-diagnostic.json')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        except OSError:
            pass

    def idle(self):
        now = self.clock()
        with self.lock:
            if self.last_observation is not None and now-self.last_observation > self.s.max_gap:
                if self.engine.tracking:
                    self.engine.update(Observation(now))
                    self.revision += 1
                if self.calibrator:
                    self.calibrator.update(Observation(now), self.engine.calibration)

    def snapshot(self):
        with self.lock:
            engine = copy(self.engine)
            engine.menu = copy(self.engine.menu)
            engine.history = SimpleNamespace(entries=tuple(self.engine.history.entries))
            calibration=copy(self.calibrator)
            return SimpleNamespace(engine=engine, calibration=calibration, message=self.message,
                                   revision=self.revision, region_changes=self.region_changes,
                                   regions_seen=sorted(self.regions_seen))

    def clear(self):
        with self.lock:
            self.engine.history.clear()
            self.engine.last_confirmation = None
            self.revision += 1

    def detector_failed(self, message):
        with self.lock:
            self.engine.reset()
            self.calibrator = None
            self.message = message
            self.revision += 1

    def set_demo(self, enabled):
        with self.lock:
            if not self.demo:
                self.real_profile = self.engine.calibration
            self.demo = enabled
            self.engine.calibration = (Calibration.paged_demo() if self.s.paged_navigation else Calibration.demo()) if enabled else self.real_profile
            self.engine.reset()
            self.calibrator = None
            self.last_observation = None
            self.message = "DEMO: mouse ou 1–4; olhe para as bordas para trocar de página." if enabled else "Pressione G para calibrar o olhar."
            self.revision += 1

    def calibrate(self, guided=True):
        with self.lock:
            if self.demo:
                self.message = "Saia do DEMO (D) para calibrar a câmera real."
            else:
                self.engine.page=0
                self.calibrator = HorizontalCalibrator(self.s) if self.s.paged_navigation else Calibrator(guided, self.s)
                self.engine.reset()
                self.message = ""
            self.revision += 1

    def gain(self, increase):
        with self.lock:
            if self.engine.calibration.mapping and self.engine.calibration.mapping.get('classifier'):
                self.message='Seleção por cartão: G para recalibrar as leituras.'
                self.revision+=1
                return
            gain = max(0.25, min(4, self.engine.calibration.gain*(1.1 if increase else 1/1.1)))
            self.engine.calibration = replace(self.engine.calibration, gain=gain)
            self.engine.reset()
            self.message = f"Sensibilidade: {gain:.2f}×"
            if not self.demo:
                self.real_profile = self.engine.calibration
                try:
                    self.real_profile.save(self.path)
                except OSError:
                    self.message += " · não foi possível salvar"
            self.revision += 1
