import numpy as np
import numpy.typing as npt

import warnings

import cv2


def estimate_vertical_shift_orb(
        bottom_incoming_image : npt.NDArray,
        top_stitched_image: npt.NDArray,
        starting_roi_xyxy: list[int], 
        horizontal_decimation : int = 1,
        n_features: int = 30,
        k1d1_precomputed : tuple[cv2.KeyPoint, npt.NDArray] | None = None,
    ) -> tuple[float, tuple[npt.NDArray[np.float64], float]]:
    """
    Estimates the vertical shift in pixels from still_image to shift_image using orb feature matching.
    Note that the maximum shift is implicitly the y-range of the starting region of interest. 


    Arguments:
        - bottom_incoming_image : shifted image to align
        - top_stitched_image : starting image to align with
        - starting_roi_xyxy: bounding box of the bottom_stitched_image to use for alignment
        - upsample_factor : fft upsampling for sub-pixel precision.
                default=1 (no upsampling).
        
    Returns:
        pixelshift : Estimated vertical shift in pixels necessary to align shift_image with still_image 
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

    orb = cv2.ORB_create(n_features) # type: ignore
    if k1d1_precomputed is None:
        k1, d1 = orb.detectAndCompute(top_stitched_image[roi_y1:roi_y2, roi_x1:roi_x2:horizontal_decimation], None)
    else:
        k1, d1 = k1d1_precomputed
    k2, d2 = orb.detectAndCompute(bottom_incoming_image[roi_y1:roi_y2, roi_x1:roi_x2:horizontal_decimation], None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(d1, d2)
    # we don't need matches to be sorted
    # matches = sorted(matches, key=lambda m: m.distance)

    if not matches:
        # TODO decide if we should return the first or
        # second image keypoints ...
        return 0.0, (k1, d1)

    dy_values = []
    for m in matches:
        p1 = np.array(k1[m.queryIdx].pt)
        p2 = np.array(k2[m.trainIdx].pt)
        dy = p1[1] - p2[1]   # vertical movement (pixels)
        dy_values.append(dy)

    # Median is robust to mismatches
    pixelshift = np.median(dy_values)
    return pixelshift, (k2, d2)
