"""Viewport nativo, respeitando a proporção e o orçamento de pixels da interface."""
import math


def drawing_size(width, height, max_pixels=3686400):
    if width < 320 or height < 240:
        width, height = 1280, 720
    scale = min(1.0, math.sqrt(max_pixels/(width*height)))

    return max(1, int(width*scale)), max(1, int(height*scale))


def screen_size():
    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        size = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return size
    except Exception:
        return 1280, 720


def viewport_size(cv2, window):
    try:
        _, _, w, h = cv2.getWindowImageRect(window)
        if w >= 320 and h >= 240:
            return w, h
    except (cv2.error, AttributeError):
        pass
    return None
