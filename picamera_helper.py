"""Captura imagens CSI em um processo separado."""
import argparse
import struct
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    from picamera2 import Picamera2
    if args.probe:
        return 0 if Picamera2.global_camera_info() else 1
    camera = Picamera2()
    try:
        camera.configure(camera.create_video_configuration(
            main={"size": (args.width, args.height), "format": "RGB888"},
            controls={"FrameRate": args.fps}, buffer_count=2))
        camera.start()
        while True:
            data = camera.capture_array("main").tobytes()
            header = struct.pack("!dI", time.perf_counter(), len(data))
            sys.stdout.buffer.write(header)
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()
    except (BrokenPipeError, KeyboardInterrupt):
        pass
    finally:
        camera.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
