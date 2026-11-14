"""Configuração portátil; os tempos são sempre em segundos."""
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path


@dataclass
class Settings:
    width: int = 640
    height: int = 480
    camera_fps: int = 30
    processing_fps: int = 20
    cv_threads: int = 2
    dwell: float = 0.25
    hysteresis: float = 0.022
    rearm: float = 0.18
    max_gap: float = 0.35
    ear_open: float = 0.28
    close_ratio: float = 0.65
    reopen_ratio: float = 0.80
    long_min: float = 0.40
    long_max: float = 1.35
    confirm_cooldown: float = 0.8
    history_limit: int = 8
    camera_mjpeg: bool = True
    display_fps: int = 30
    preview_enabled: bool = True
    fullscreen: bool = True
    confirmation_seconds: float = 5.0
    fixation_grace: float = 0.4
    repeat_release: float = 0.8
    ui_max_pixels: int = 3686400
    tracking_median: int = 5
    tracking_window_seconds: float = 0.24
    filter_cutoff: float = 0.35
    filter_beta: float = 1.5
    row_switch: float = 0.35
    column_switch: float = 0.14
    calibration_collect: float = 1.5
    calibration_settle: float = 1.2
    calibration_passes: int = 2
    paged_navigation: bool = True
    navigation_seconds: float = 1.5

    def validate(self):
        for key, value in asdict(self).items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f"{key}: valor não finito")
        for key in ("width", "height", "camera_fps", "processing_fps", "display_fps", "cv_threads", "history_limit", "ui_max_pixels"):
            if type(getattr(self, key)) is not int:
                raise ValueError(f"{key}: use um número inteiro")
        if not (160 <= self.width <= 1280 and 120 <= self.height <= 720):
            raise ValueError("Resolução permitida: 160x120 a 1280x720")
        if not (1 <= self.processing_fps <= 60 and 1 <= self.camera_fps <= 60 and 1 <= self.display_fps <= 60
                and 1 <= self.cv_threads <= 4):
            raise ValueError("FPS deve estar entre 1 e 60; threads, entre 1 e 4")
        if not (0 < self.close_ratio < self.reopen_ratio < 1 and 0.05 < self.ear_open < 0.8):
            raise ValueError("Limiares EAR inválidos")
        if not (0 < self.dwell <= 3 and 0 <= self.hysteresis < 0.1
                and 0 < self.rearm <= 2 and 0.05 <= self.max_gap <= 0.5):
            raise ValueError("Tempos de seleção/rastreamento inválidos")
        if not (0.2 <= self.long_min < self.long_max <= 2 and 0.2 <= self.confirm_cooldown <= 5):
            raise ValueError("Tempos de confirmação inválidos")
        if not 1 <= self.history_limit <= 50 or type(self.camera_mjpeg) is not bool or type(self.preview_enabled) is not bool:
            raise ValueError("Histórico ou formato de câmera inválido")
        if (type(self.fullscreen) is not bool or not 1 <= self.confirmation_seconds <= 30
                or not 0 <= self.fixation_grace <= 0.8 or not 0.2 <= self.repeat_release <= 3):
            raise ValueError("Tela cheia ou tempos de fixação inválidos")
        if not 307200 <= self.ui_max_pixels <= 8294400:
            raise ValueError("Orçamento da interface deve estar entre 307200 e 8294400 pixels")
        if (type(self.tracking_median) is not int or self.tracking_median not in (3,5,7,9,11)
                or not .1 <= self.tracking_window_seconds <= .5
                or not .2 <= self.filter_cutoff <= 3 or not 0 <= self.filter_beta <= 2
                or not .05 <= self.row_switch <= 1 or not .05 <= self.column_switch <= 1
                or not 1.2 <= self.calibration_collect <= 5 or not 1 <= self.calibration_settle <= 3):
            raise ValueError('Parâmetros de estabilidade inválidos')
        if type(self.calibration_passes) is not int or self.calibration_passes not in (1,2):
            raise ValueError('Passagens da calibração devem ser 1 ou 2')
        if type(self.paged_navigation) is not bool or not .8 <= self.navigation_seconds <= 4:
            raise ValueError('Navegação por páginas inválida')
        return self

    @classmethod
    def load(cls, path=None, mode='lite'):
        if path is None:
            return cls().validate()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or set(data) - {f.name for f in fields(cls)}:
            raise ValueError("Configuração contém campos desconhecidos")
        return cls(**data).validate()
