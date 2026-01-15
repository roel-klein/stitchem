import numpy as np
import numpy.typing as npt
from numba import njit # type: ignore 
from functools import cache

@cache
def create_grayscale_lut():
    # Define the BGR to grayscale conversion formula as a Look-Up Table (LUT)
    # We use the formula: Gray = 0.114 * B + 0.587 * G + 0.299 * R
    bgr_factors = [0.114, 0.587, 0.299]
    # Create a LUT for each color channel (B, G, R)
    inputs = np.arange(0, 256, dtype=np.uint8)
    
    lut = []
    for factor in bgr_factors:
        lut.append((inputs * factor).astype(np.uint8))
    return lut

def bgr2graylut(img):
    LUT = create_grayscale_lut()
    img = LUT[0][img[:,:, 0]] + LUT[1][img[:,:, 1]] + LUT[2][img[:,:, 2]]
    return img

@njit(cache=True)
def bgr2gray(img, dtype=np.uint8):
    """
    A faster alternative for cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    """
    # Y=0.299*R + 0.587*G + 0.114*B
    return (0.299 * img[:,:, 2] + 0.587 * img[:,:, 1] + 0.114 * img[:,:, 0]).astype(dtype)

def preprocess(
        bgr_img : npt.NDArray, 
        horizontal_decimation : int, 
        starting_roi_xyxy : list[int], 
        to_grayscale=True, 
        dtype=np.uint8,
        ) -> tuple[npt.NDArray, list[int]]:
    updated_roi_xyxy = [0, starting_roi_xyxy[1], (starting_roi_xyxy[2] - starting_roi_xyxy[0]) // horizontal_decimation, starting_roi_xyxy[3]]
    if to_grayscale:
        return bgr2gray(bgr_img[:, starting_roi_xyxy[0]:starting_roi_xyxy[2]:horizontal_decimation], dtype=dtype), updated_roi_xyxy
    else:
        return (bgr_img[:, starting_roi_xyxy[0]:starting_roi_xyxy[2]:horizontal_decimation]).astype(dtype), updated_roi_xyxy

if __name__ == "__main__":
    import cv2
    import time
    
    # Create a test image
    img = np.random.randint(0, 256, (10000, 1000, 3), dtype=np.uint8)
    
    # Warm up the JIT compilation
    _ = bgr2gray(img)
    
    # Benchmark bgr2gray
    n_iterations = 100
    start = time.perf_counter()
    for _ in range(n_iterations):
        result_custom = bgr2gray(img)
    time_custom = (time.perf_counter() - start) / n_iterations
    
    # Benchmark cv2.cvtColor
    start = time.perf_counter()
    for _ in range(n_iterations):
        result_cv2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    time_cv2 = (time.perf_counter() - start) / n_iterations

    # Benchmark bgr2graylut
    create_grayscale_lut()
    start = time.perf_counter()
    for _ in range(n_iterations):
        result_lut = bgr2graylut(img)
    time_lut = (time.perf_counter() - start) / n_iterations
    
    # Print results
    print(f"Image shape: {img.shape}")
    print(f"bgr2gray:     {time_custom * 1000:.3f} ms")
    print(f"cv2.cvtColor: {time_cv2 * 1000:.3f} ms")
    print(f"bgr2graylut: {time_lut * 1000:.3f} ms")
    print(f"Speedup:      {time_cv2 / time_custom:.2f}x")
    
    # Verify results are similar
    diff = np.abs(result_custom.astype(np.float32) - result_cv2.astype(np.float32)).mean()
    print(f"Mean difference: {diff:.4f}")
