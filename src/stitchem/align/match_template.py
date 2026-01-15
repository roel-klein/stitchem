import numpy as np
import numpy.typing as npt

import warnings

import cv2

def estimate_vertical_shift_match_template(
        bottom_incoming_image : npt.NDArray, 
        top_stitched_image: npt.NDArray, 
        starting_roi_xyxy: list[int], 
        max_shift : int,
        method = cv2.TM_CCOEFF,
    ) -> tuple[int, npt.NDArray]:
    """
    Estimates the vertical shift in pixels from top_stitched_image to bottom_incoming_image.
    Arguments:
        - bottom_incoming_image : shifted image to align, grayscale np.uint8
        - top_stitched_image : starting image to align with, grayscale np.uint8
        - starting_roi_xyxy: bounding box of the bottom_stitched_image to use for alignment
        - max_shift : maximum vertical shift in pixels
    Returns:
        pixelshift : Estimated number pixels of that should be added from incoming image. 
        errors : For each pixelshift, the sum of the MAE.
    """
    height = bottom_incoming_image.shape[0]
    max_shift = min(max_shift, height)

    roi_x1 = starting_roi_xyxy[0]
    roi_y1 = starting_roi_xyxy[1]
    roi_x2 = starting_roi_xyxy[2]
    roi_y2 = starting_roi_xyxy[3]

    if roi_y1 < max_shift:
        warnings.warn(
            f"The region of interest y-position {roi_y1} is smaller than maximum shift {max_shift}. Adjusting y-position to {max_shift}, to make it possible to check previous pixel lines. Adjusting either max_shift or still_box_xyxy is recommended to prevent possible runtimeerrors.",
            stacklevel=2,
        )
        # further shift starting roi y-position if necessary
        # to make sure we can look at the max_shift pixels before it
        roi_y2 += max_shift - roi_y1     
        roi_y1 += max_shift - roi_y1
        if roi_y2 > height:
            raise RuntimeError("Bounding box exceeded image border, reduce max shift or roi box size")
    
    # convert roi indices to be negative, relative to image end
    # this way we can use the same roi indices for a longer stitched image and a shorter incoming image
    roi_y1 -= bottom_incoming_image.shape[0]
    roi_y2 -= bottom_incoming_image.shape[0]

    # Apply template Matching
    search = bottom_incoming_image[roi_y1-max_shift:roi_y2, roi_x1:roi_x2]
    template = top_stitched_image[roi_y1:roi_y2, roi_x1:roi_x2]
    res = cv2.matchTemplate(search,template,method)
 
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, lower is better (error metric)
    # for other, higher is better(similarity metric)
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        pixelshift = int(np.argmin(res[::-1, 0]))
    else:
        pixelshift = int(np.argmax(res[::-1, 0]))
    return pixelshift, res


def estimate_vertical_shift_match_template_twostage(
        bottom_incoming_image : npt.NDArray, 
        top_stitched_image: npt.NDArray, 
        starting_roi_xyxy: list[int], 
        max_shift : int,
        first_stage_decimation : int = 4,
        method = cv2.TM_SQDIFF,
    ) -> tuple[int, npt.NDArray]:
    """
    Estimates the vertical shift in pixels from top_stitched_image to bottom_incoming_image.
    Arguments:
        - bottom_incoming_image : shifted image to align, grayscale np.uint8
        - top_stitched_image : starting image to align with
        - starting_roi_xyxy: bounding box of the bottom_stitched_image to use for alignment
        - max_shift : maximum vertical shift in pixels
        - first_stage_decimation : first-stage subsampling factor in vertical direction
    Returns:
        pixelshift : Estimated number pixels of that should be added from incoming image. 
        errors : For each pixelshift, the sum of the MAE.
    """
    height = bottom_incoming_image.shape[0]
    max_shift = min(max_shift, height)

    roi_x1 = starting_roi_xyxy[0]
    roi_y1 = starting_roi_xyxy[1]
    roi_x2 = starting_roi_xyxy[2]
    roi_y2 = starting_roi_xyxy[3]

    if roi_y1 < max_shift:
        warnings.warn(
            f"The region of interest y-position {roi_y1} is smaller than maximum shift {max_shift}. Adjusting y-position to {max_shift}, to make it possible to check previous pixel lines. Adjusting either max_shift or still_box_xyxy is recommended to prevent possible runtimeerrors.",
            stacklevel=2,
        )
        # further shift starting roi y-position if necessary
        # to make sure we can look at the max_shift pixels before it
        roi_y2 += max_shift - roi_y1     
        roi_y1 += max_shift - roi_y1
        if roi_y2 > height:
            raise RuntimeError("Bounding box exceeded image border, reduce max shift or roi box size")
    
    # convert roi indices to be negative, relative to image end
    # this way we can use the same roi indices for a longer stitched image and a shorter incoming image
    roi_y1 -= bottom_incoming_image.shape[0]
    roi_y2 -= bottom_incoming_image.shape[0]

    # Stage 1 : Coarse
    search = bottom_incoming_image[roi_y1-max_shift:roi_y2:first_stage_decimation, roi_x1:roi_x2]
    template = top_stitched_image[roi_y1:roi_y2:first_stage_decimation, roi_x1:roi_x2]

    res = cv2.matchTemplate(search,template,method)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, lower is better (error metric)
    # for other, higher is better(similarity metric)
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        pixelshift = int(np.argmin(res[::-1, 0]) * first_stage_decimation)
    else:
        pixelshift = int(np.argmax(res[::-1, 0]) * first_stage_decimation)

    # Stage 2 : Pixel-precise
    new_y1 = max(roi_y1 - pixelshift - first_stage_decimation, -bottom_incoming_image.shape[0]) + 1
    new_y2 = min(roi_y2 - pixelshift + first_stage_decimation, roi_y2)
    search = bottom_incoming_image[new_y1:new_y2, roi_x1:roi_x2]
    template = top_stitched_image[roi_y1:roi_y2, roi_x1:roi_x2]

    res = cv2.matchTemplate(search,template,method)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, lower is better (error metric)
    # for other, higher is better(similarity metric)
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        pixelshift = int(np.argmin(res[::-1, 0]))
    else:
        pixelshift = int(np.argmax(res[::-1, 0]))
    return pixelshift, res