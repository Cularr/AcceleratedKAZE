"""Writes the OpenCV reference files. Requires opencv-python-headless==4.12.0.88."""
from pathlib import Path
import struct
import cv2
import numpy as np

if not cv2.__version__.startswith("4.12."):
    raise SystemExit(f"Reference must be OpenCV 4.12, found {cv2.__version__}")

folder = Path(__file__).resolve().parent
record = struct.Struct("<5f2i61s")
detector = cv2.AKAZE_create()
graf = cv2.imread(str(folder / "graf1.png"), 0)
box = cv2.imread(str(folder / "box.png"), 0)
noise = np.random.default_rng(48).integers(0, 256, (193, 257), dtype=np.uint8)
noise.tofile(folder / "noise-257x193.gray")


def akaze(name, image):
    points, descriptors = detector.detectAndCompute(image, None)
    rows = zip(points, [] if descriptors is None else descriptors)
    data = b"".join(record.pack(*p.pt, p.size, p.angle, p.response, p.octave, p.class_id, d.tobytes()) for p, d in rows)
    (folder / f"{name}.akaze").write_bytes(data)
    return points


akaze("box", box)
akaze("graf1", graf)
akaze("graf1-797x637", np.ascontiguousarray(graf[:637, :797]))
akaze("noise-257x193", noise)
akaze("blank", np.zeros((128, 128), np.uint8))
for width, height in ((200, 60), (200, 59), (200, 58), (60, 200), (59, 200)):
    candidates = [np.random.default_rng(seed).integers(0, 256, (height, width), dtype=np.uint8) for seed in range(6)]
    narrow = max(candidates, key=lambda image: len(detector.detect(image)))
    narrow.tofile(folder / f"narrow-{width}x{height}.gray")
    akaze(f"narrow-{width}x{height}", narrow)

y, x = np.mgrid[0:121, 0:161]
color = np.stack([(x * 255 // 160), (y * 255 // 120), ((x + y) * 255 // 280)], axis=2).astype(np.int32)
color = np.clip(color + np.random.default_rng(161).integers(-24, 25, color.shape), 0, 255).astype(np.uint8)
cv2.imwrite(str(folder / "color.jpg"), color, [cv2.IMWRITE_JPEG_QUALITY, 90])
cv2.imwrite(str(folder / "color.png"), color)
for name in ("color.jpg", "color.png"):
    decoded = cv2.imdecode(np.fromfile(folder / name, np.uint8), cv2.IMREAD_COLOR)
    cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY).tofile(folder / f"{name}.gray")
