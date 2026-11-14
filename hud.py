"""HUD OpenCV com texto UTF-8 em cache, sem navegador ou GPU."""
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from core import LABELS
from horizontal_calibration import card_bounds

BG = (20, 17, 12)
PANEL = (37, 30, 22)
LINE = (68, 58, 45)
WHITE = (245, 240, 227)
MUTED = (170, 159, 138)
TEAL = (196, 224, 73)
GREEN = (123, 213, 74)
AMBER = (80, 189, 248)


@lru_cache(maxsize=16)
def font(size, bold=False):
    candidates = [Path("C:/Windows/Fonts/seguisb.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
                  Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
                       "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


@lru_cache(maxsize=256)
def text_tile(text, size, color, bold):
    f = font(size, bold)
    bbox = f.getbbox(text)
    tile = Image.new("RGBA", (max(1, bbox[2]+2), max(1, bbox[3]+4)), (0, 0, 0, 0))
    ImageDraw.Draw(tile).text((0, 0), text, font=f, fill=(*color[::-1], 255))
    rgba = np.asarray(tile)
    return rgba[:, :, :3][:, :, ::-1].copy(), rgba[:, :, 3:4]/255.0


def text(canvas, message, x, y, size=18, color=WHITE, bold=False, center=False):
    pixels, alpha = text_tile(str(message), size, tuple(color), bold)
    h, w = pixels.shape[:2]
    if center:
        x -= w//2
    x, y = int(x), int(y)
    if x < 0 or y < 0 or x >= canvas.shape[1] or y >= canvas.shape[0]:
        return
    h, w = min(h, canvas.shape[0]-y), min(w, canvas.shape[1]-x)
    region = canvas[y:y+h, x:x+w]
    region[:] = (pixels[:h, :w]*alpha[:h, :w]+region*(1-alpha[:h, :w])).astype(np.uint8)


class HUD:
    """Layout em proporções; o OpenCV amplia o canvas para preencher o monitor."""
    width, height = 1280, 720

    def __init__(self, width=1280, height=720, max_pixels=3686400):
        self.max_pixels = max_pixels
        self._panel_key = self._panel = self._preview = self._source_image = None
        self._mirror = None
        self._telemetry = self._telemetry_tick = None
        self.panel_builds = 0
        self.paged_layout=False
        self.resize(width, height)

    def resize(self, width, height, force=False):
        from display import drawing_size
        width, height = drawing_size(width, height, self.max_pixels)
        if not force and (width, height) == (getattr(self, 'width', None), getattr(self, 'height', None)) and hasattr(self, 'grid_rect'):
            return False
        self.width, self.height = width, height
        self.scale = min(width/1280, height/720)
        self.grid_rect = (int(width*.025), int(height*.43), int(width*.975), int(height*.90))
        self.preview_rect = (int(width*.025), int(height*.13), int(width*.20), int(height*.34))
        self.status_rect = (int(width*.215), int(height*.13), int(width*.515), int(height*.34))
        self.history_rect = (int(width*.53), int(height*.13), int(width*.975), int(height*.34))
        hx, hy, hr, hb = self.history_rect
        self.clear_rect = (hr-self.sz(82), hy+self.sz(9), hr-self.sz(10), hy+self.sz(35))
        self.card_rects = []
        gx, gy, gr, gb = self.grid_rect
        for i in range(len(LABELS)):
            if self.paged_layout:
                col=i%4
                left,right=card_bounds(col)
                self.card_rects.append((int(gx+left*(gr-gx)),gy+int((gb-gy)*.018),
                                       int(gx+right*(gr-gx)),gb-int((gb-gy)*.018)))
                continue
            col, row = i%4, i//4
            self.card_rects.append((int(gx+(col/4+.009)*(gr-gx)), int(gy+(row/2+.018)*(gb-gy)),
                                    int(gx+((col+1)/4-.009)*(gr-gx)), int(gy+((row+1)/2-.018)*(gb-gy))))
        self._panel_key = self._preview = self._source_image = None
        self._telemetry_tick = None
        return True

    def sz(self, value):
        return max(1, round(value*self.scale))

    def label(self, canvas, message, x, y, size=18, color=WHITE, bold=False, center=False, max_width=None):
        size = max(9, self.sz(size))
        if max_width is not None:
            while size > 8 and font(size, bold).getlength(str(message)) > max_width:
                size -= 1
        text(canvas, message, int(x), int(y), size, color, bold, center)

    def grid_point(self, x, y):
        l, t, r, b = self.grid_rect
        return int(l+x*(r-l)), int(t+y*(b-t))

    def demo_position(self, x, y):
        l, t, r, b = self.grid_rect
        return (x-l)/(r-l), (y-t)/(b-t)

    def clear_hit(self, x, y):
        l, t, r, b = self.clear_rect
        return l <= x <= r and t <= y <= b

    def clear_history_at(self, engine, x, y):
        if not self.clear_hit(x, y):
            return False
        engine.history.clear()
        engine.last_confirmation = None
        return True

    def render(self, image, engine, camera_status, vision_status,
               demo=False, mirrored=True, fps=0, latency=0, temperature=None,
               calibration=None, message="", camera_fps=0, now=0,
               preview_visible=True, display_fps=None):
        paged=getattr(engine,'paged_ui',False) or bool(engine.calibration.mapping and engine.calibration.mapping.get('horizontal')) or type(calibration).__name__=='HorizontalCalibrator'
        if paged!=self.paged_layout:
            self.paged_layout=paged
            self.resize(self.width,self.height,force=True)
        last = engine.last_confirmation
        recent = last is not None and 0 <= now-last.timestamp < 2
        key = (demo, engine.tracking, engine.menu.selected, engine.blocked,engine.page,paged,
               tuple((e.clock, e.label) for e in engine.history.entries),
               None if calibration is None else (calibration.index, calibration.error),
               message, engine.quality_message, vision_status, camera_status, preview_visible,
               None if last is None else (last.block, last.timestamp, recent))
        if key != self._panel_key:
            self._panel = self._compose_panel(engine, camera_status, vision_status, demo,
                                              calibration, message, recent, preview_visible)
            self._panel_key = key
            self.panel_builds += 1
        canvas = self._panel.copy()
        if paged and engine.nav_side:
            gx,gy,gr,gb=self.grid_rect
            left=gx if engine.nav_side=='previous' else int(gx+.92*(gr-gx))
            right=int(gx+.08*(gr-gx)) if engine.nav_side=='previous' else gr
            cv2.rectangle(canvas,(left,gb-self.sz(10)),(left+int((right-left)*engine.nav_progress),gb-self.sz(4)),TEAL,-1)
        x0, y0, right, bottom = self.preview_rect
        pad = self.sz(5)
        w, h = right-x0-2*pad, bottom-y0-2*pad
        if image is not None and preview_visible:
            if image is not self._source_image or mirrored != self._mirror:
                ih, iw = image.shape[:2]
                scale = min(w/iw, h/ih)
                self._preview = cv2.resize(image, (max(1, int(iw*scale)), max(1, int(ih*scale))), interpolation=cv2.INTER_AREA)
                if mirrored:
                    self._preview = cv2.flip(self._preview, 1)
                self._source_image, self._mirror = image, mirrored
            rh, rw = self._preview.shape[:2]
            sx, sy = x0+(right-x0-rw)//2, y0+(bottom-y0-rh)//2
            canvas[sy:sy+rh, sx:sx+rw] = self._preview

        i = engine.fixation
        if i is not None and calibration is None and engine.blocked != i:
            l, t, r, b = self.card_rects[i]
            cv2.rectangle(canvas, (l+self.sz(8), b-self.sz(15)), (r-self.sz(8), b-self.sz(9)), LINE, -1)
            end = l+self.sz(8)+int((r-l-2*self.sz(8))*engine.confirm_progress)
            cv2.rectangle(canvas, (l+self.sz(8), b-self.sz(15)), (end, b-self.sz(9)), TEAL, -1)
            self.label(canvas, f"{engine.fixation_elapsed:.1f} / {engine.s.confirmation_seconds:g} s".replace('.', ','),
                       (l+r)//2, b-self.sz(44), 14, TEAL, center=True)
        if calibration:
            cx, cy = self.grid_point(*calibration.position)
            cv2.circle(canvas, (cx, cy), self.sz(17), AMBER, 3, cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), self.sz(4)//2, WHITE, -1)
        elif engine.tracking and engine.calibration.mapping is not None:
            cx, cy = self.grid_point(float(np.clip(engine.x,0,1)),float(np.clip(engine.y,-.6,1.1)))
            if 0 <= cx < self.width and 0 <= cy < self.height:
                cv2.circle(canvas, (cx, cy), self.sz(10), TEAL, 2, cv2.LINE_AA)
                cv2.circle(canvas, (cx, cy), 2, WHITE, -1)

        tick = int(now*2)
        if self._telemetry is None or tick != self._telemetry_tick:
            self._telemetry = np.full((self.sz(24), self.width, 3), BG, np.uint8)
            temp = f" · {temperature:.0f} °C" if temperature is not None else ""
            screen = f" · tela {display_fps:.0f}" if display_fps is not None else ""
            self.label(self._telemetry, f"Câmera {camera_fps:.0f} · visão {fps:.0f}{screen} FPS · {latency:.0f} ms{temp} · Dados locais",
                       self.width*.025, 0, 12, MUTED, max_width=self.width*.95)
            self._telemetry_tick = tick
        ty = int(self.height*.96)
        th = min(self._telemetry.shape[0], self.height-ty)
        canvas[ty:ty+th] = self._telemetry[:th]
        return canvas

    def _compose_panel(self, engine, camera_status, vision_status, demo, calibration, message, recent, preview_visible):
        c = np.full((self.height, self.width, 3), BG, np.uint8)
        w, h = self.width, self.height
        self.label(c, "EyeBridge", w*.025, h*.015, 32, WHITE, True)
        self.label(c, f"OLHE POR {engine.s.confirmation_seconds:g} SEGUNDOS PARA CONFIRMAR", w*.025, h*.073, 12, MUTED, True, max_width=w*.64)
        self.label(c, "DEMONSTRAÇÃO" if demo else "USB · "+getattr(self,'mode','normal').upper(), w*.73, h*.037, 13, AMBER if demo else TEAL, True, max_width=w*.245)
        for rect in (self.preview_rect, self.status_rect, self.history_rect):
            l,t,r,b = rect
            cv2.rectangle(c, (l,t), (r,b), PANEL, -1)
        l,t,r,b = self.preview_rect
        if not preview_visible:
            self.label(c, "VÍDEO OCULTO", (l+r)//2, (t+b)//2-self.sz(12), 13, MUTED, center=True, max_width=r-l-10)
        l,t,r,b = self.status_rect
        title = "CALIBRAÇÃO" if calibration else ("OLHAR DETECTADO" if engine.tracking else "AGUARDANDO ROSTO")
        self.label(c, title, l+self.sz(14), t+self.sz(11), 13, TEAL if engine.tracking else AMBER, True, max_width=r-l-24)
        selected = engine.menu.selected
        status = "Fixe o ponto amarelo" if calibration else (LABELS[selected] if selected is not None else "Olhe para um cartão")
        if recent:
            status = "Confirmado: "+engine.last_confirmation.label
        self.label(c, status, l+self.sz(14), t+self.sz(42), 24, GREEN if recent else WHITE, True, max_width=r-l-28)
        hint = ("Olhe para outro cartão para repetir" if engine.blocked is not None else "Piscadas naturais apenas pausam")
        self.label(c, hint, l+self.sz(14), b-self.sz(30), 12, MUTED, max_width=r-l-28)
        l,t,r,b = self.history_rect
        self.label(c, "HISTÓRICO", l+self.sz(14), t+self.sz(11), 13, MUTED, True)
        cl,ct,cr,cb = self.clear_rect
        cv2.rectangle(c, (cl,ct),(cr,cb),LINE,1)
        self.label(c, "CLEAR", (cl+cr)//2, ct+self.sz(2), 12, WHITE, True, center=True)
        entries = list(engine.history.entries)[:6]
        for j, entry in enumerate(entries):
            col,row = j//3, j%3
            x = l+self.sz(14)+col*(r-l)//2
            yy = t+self.sz(43+28*row)
            self.label(c, entry.clock, x, yy, 11, MUTED)
            self.label(c, entry.label, x+self.sz(67), yy-self.sz(1), 14, WHITE, j==0, max_width=(r-l)//2-self.sz(85))
        if not entries:
            self.label(c, "As confirmações aparecerão aqui", l+self.sz(14), t+self.sz(58), 14, MUTED, max_width=r-l-28)
        banner = (f"{calibration.index+1}/{len(calibration.targets)} · OLHE PARA {calibration.target}" if calibration else
                  (engine.quality_message or message or ("Escolha registrada no histórico" if recent else "Olhe por 5 s · Bordas ANT./PRÓX. por 1,5 s trocam a página")))
        if calibration and calibration.error:
            banner = calibration.error
        self.label(c, banner, w*.5, h*.365, 17, AMBER if calibration or message else WHITE, True, center=True, max_width=w*.94)
        if self.paged_layout:
            gx,gy,gr,gb=self.grid_rect
            for l,r,label,enabled in [(gx,int(gx+.08*(gr-gx)),'ANT.',engine.page==1),(int(gx+.92*(gr-gx)),gr,'PRÓX.',engine.page==0)]:
                cv2.rectangle(c,(l,gy),(r,gb),PANEL,-1)
                self.label(c,label,(l+r)//2,(gy+gb)//2,14,TEAL if enabled else MUTED,True,center=True,max_width=r-l-8)
            self.label(c,f'PÁGINA {engine.page+1}/2',w*.5,h*.405,12,TEAL,True,center=True)
        for i,(l,t,r,b) in enumerate(self.card_rects):
            if self.paged_layout and i//4!=engine.page:continue
            selected = engine.menu.selected == i
            confirmed = recent and engine.last_confirmation.block == i
            fill = (48,63,28) if confirmed else ((49,51,28) if selected else PANEL)
            border = GREEN if confirmed else (TEAL if selected else LINE)
            if calibration and calibration.order[calibration.index] == i:
                border = AMBER
            cv2.rectangle(c,(l,t),(r,b),fill,-1)
            cv2.rectangle(c,(l,t),(r,b),border,2 if selected else 1)
            self.label(c,f"{i+1:02d}",l+self.sz(11),t+self.sz(6),11,MUTED)
            self.label(c,LABELS[i].upper(),(l+r)//2,t+(b-t)*.27,27,WHITE,True,center=True,max_width=r-l-self.sz(24))
            if engine.blocked == i:
                self.label(c,"CONFIRMADO",(l+r)//2,b-self.sz(36),12,GREEN,True,center=True,max_width=r-l-16)
        self.label(c,"G  Calibrar   H  Limpar   V  Vídeo   F  Tela cheia   D  Demo   Q / Esc  Sair",
                   w*.025,h*.92,12,MUTED,max_width=w*.95)
        return c
