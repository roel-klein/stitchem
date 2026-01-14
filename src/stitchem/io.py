from pathlib import Path
import cv2

def img_generator(img_dir : Path, every_n_frames=1, start=0):
    for i, img_path in enumerate(sorted(img_dir.glob("*"))):
        if i < start:
            continue
        if i % every_n_frames == 0:
            yield cv2.imread(str(img_path))
