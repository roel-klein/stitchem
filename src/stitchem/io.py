from pathlib import Path
from typing import Generator

import cv2
import numpy.typing as npt

def img_generator(img_dir : Path, every_n_frames=1, start=0) -> Generator[npt.NDArray, None, None]:
    for i, img_path in enumerate(sorted(img_dir.glob("*"))):
        if i < start:
            continue
        if i % every_n_frames == 0:
            img = cv2.imread(str(img_path))
            if img is None:
                raise IOError(f"Error loading image at path {img_path}")
            yield img
