import numpy as np
import numpy.typing as npt
from skimage.registration import phase_cross_correlation

from stitchem.conversion import bgr2gray
import warnings

import cv2


def estimate_shift_phase_cross_correlation_skimage(
        bottom_incoming_image : npt.NDArray,
        top_stitched_image: npt.NDArray,
        starting_roi_xyxy: list[int], 
        upsample_factor: int = 1,
    ) -> tuple[npt.NDArray[np.float64], float]:
    """
    Estimates the vertical shift in pixels from still_image to shift_image using phase cross correlation.
    Note that the maximum shift is implicitly the y-range of the starting region of interest. 

    Arguments:
        - bottom_incoming_image : shifted image to align, grayscale np.float32
        - top_stitched_image : starting image to align with, grayscale np.float32
        - starting_roi_xyxy: bounding box of the bottom_stitched_image to use for alignment
        - upsample_factor : fft upsampling for sub-pixel precision.
                default=1 (no upsampling).
        
    Returns:
        pixelshift_yx : Estimated shift in pixels necessary to align shift_image with still_image, both in y and x direction 
        error :  Translation invariant normalized RMS error
    """
    max_shift = bottom_incoming_image.shape[0]

    roi_x1 = starting_roi_xyxy[0]
    roi_y1 = starting_roi_xyxy[1]
    roi_x2 = starting_roi_xyxy[2]
    roi_y2 = starting_roi_xyxy[3]

    if roi_y2 > max_shift:
        warnings.warn(
            f"The region of interest y-position {roi_y2} is larger than the incoming image height {max_shift}. Adjusting y2-position to {max_shift}. You should change your settings.",
            stacklevel=2,
        )
        # further shift starting roi y-position if necessary
        # to make sure we can look at the max_shift pixels before it
        roi_y2 += max_shift - roi_y2     
        roi_y1 += max_shift - roi_y2
        if roi_y1 < 0:
            raise RuntimeError("Bounding box exceeded image border, reduce roi box size")
    
    # convert roi indices to be negative, relative to image end
    # this way we can use the same roi indices for a longer stitched image and a shorter incoming image
    roi_y1 -= bottom_incoming_image.shape[0]
    roi_y2 -= bottom_incoming_image.shape[0]

    pixelshift_yx, error, _ = phase_cross_correlation(
            top_stitched_image[roi_y1:roi_y2, roi_x1:roi_x2], 
            bottom_incoming_image[roi_y1:roi_y2, roi_x1:roi_x2], 
            normalization=None,
            upsample_factor=upsample_factor,
            disambiguate=False,
        )
    return pixelshift_yx, error


def estimate_shift_phase_cross_correlation_opencv(
        bottom_incoming_image : npt.NDArray,
        top_stitched_image: npt.NDArray,
        starting_roi_xyxy: list[int], 
    ) -> tuple[npt.NDArray[np.float64], float]:
    """
    Estimates the vertical shift in pixels from still_image to shift_image using phase cross correlation.
    Note that the maximum shift is implicitly the y-range of the starting region of interest. 


    Arguments:
        - bottom_incoming_image : shifted image to align, grayscale np.float32
        - top_stitched_image : starting image to align with, grayscale np.float32
        - starting_roi_xyxy: bounding box of the bottom_stitched_image to use for alignment
    Returns:
        pixelshift_yx : Estimated shift in pixels necessary to align shift_image with still_image, both in y and x direction 
        error :  Translation invariant normalized RMS error
    """
    max_shift = bottom_incoming_image.shape[0]

    roi_x1 = starting_roi_xyxy[0]
    roi_y1 = starting_roi_xyxy[1]
    roi_x2 = starting_roi_xyxy[2]
    roi_y2 = starting_roi_xyxy[3]

    if roi_y2 > max_shift:
        warnings.warn(
            f"The region of interest y-position {roi_y2} is larger than the incoming image height {max_shift}. Adjusting y2-position to {max_shift}. You should change your settings.",
            stacklevel=2,
        )
        # further shift starting roi y-position if necessary
        # to make sure we can look at the max_shift pixels before it
        roi_y2 += max_shift - roi_y2     
        roi_y1 += max_shift - roi_y2
        if roi_y1 < 0:
            raise RuntimeError("Bounding box exceeded image border, reduce roi box size")
    
    # convert roi indices to be negative, relative to image end
    # this way we can use the same roi indices for a longer stitched image and a shorter incoming image
    roi_y1 -= bottom_incoming_image.shape[0]
    roi_y2 -= bottom_incoming_image.shape[0]

    # Extract ROIs and convert to float32 for OpenCV
    roi1 = top_stitched_image[roi_y1:roi_y2, roi_x1:roi_x2]
    roi2 = bottom_incoming_image[roi_y1:roi_y2, roi_x1:roi_x2]
    
    # cv2.phaseCorrelate returns (shift_xy, response)
    # Note: OpenCV returns (x, y) while we need (y, x)
    shift_xy, response = cv2.phaseCorrelate(roi2, roi1)
    
    # Convert from (x, y) to (y, x) and response to error-like metric
    pixelshift_yx = np.array([shift_xy[1], shift_xy[0]])
    error = 1.0 - response  # Convert response (quality) to error
    
    return pixelshift_yx, error

    