import numpy as np
import numpy.typing as npt

import warnings

import cv2
from stitchem.conversion import bgr2gray



def estimate_vertical_shift_match_template(
        bottom_incoming_image : npt.NDArray, 
        top_stitched_image: npt.NDArray, 
        starting_roi_xyxy: list[int], 
        max_shift : int,
        horizontal_decimation : int = 1,
        method = cv2.TM_CCOEFF,
    ) -> tuple[int, npt.NDArray]:
    """
    Estimates the vertical shift in pixels from top_stitched_image to bottom_incoming_image.
    Arguments:
        - bottom_incoming_image : shifted image to align
        - top_stitched_image : starting image to align with
        - starting_roi_xyxy: bounding box of the bottom_stitched_image to use for alignment
        - max_shift : maximum vertical shift in pixels
        - horizontal_decimation : subsampling factor in horizontal direction
    Returns:
        pixelshift : Estimated number pixels of that should be added from incoming image. 
        errors : For each pixelshift, the sum of the MAE.
    """
    bottom_incoming_image = bottom_incoming_image[:,::horizontal_decimation]
    top_stitched_image = top_stitched_image[:,::horizontal_decimation]
    if len(top_stitched_image.shape) == 3:
        top_stitched_image = bgr2gray(top_stitched_image, dtype=np.uint8)
    if len(bottom_incoming_image.shape) == 3: 
        bottom_incoming_image  = bgr2gray(bottom_incoming_image, dtype=np.uint8)

    height = bottom_incoming_image.shape[0]
    max_shift = min(max_shift, height)


    roi_x1 = starting_roi_xyxy[0] // horizontal_decimation
    roi_y1 = starting_roi_xyxy[1]
    roi_x2 = starting_roi_xyxy[2] // horizontal_decimation
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
 
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        # top_left = min_loc
        pixelshift = int(np.argmin(res[::-1, 0]))
        # top_5_idxs = np.argsort(res[::-1, 0])[:5]
    else:
        # top_left = max_loc
        pixelshift = int(np.argmax(res[::-1, 0]))
        # top_5_idxs = np.argsort(res[::-1, 0])[-5::-1]

    # top_5_values = res[::-1, 0][top_5_idxs]
    # print(f"{pixelshift=} {top_5_idxs=}, {top_5_values=}")
    return pixelshift, res


def estimate_vertical_shift_match_template_twostage(
        bottom_incoming_image : npt.NDArray, 
        top_stitched_image: npt.NDArray, 
        starting_roi_xyxy: list[int], 
        max_shift : int,
        horizontal_decimation : int = 1,
        first_stage_decimation : int = 4,
        method = cv2.TM_SQDIFF,
    ) -> tuple[int, npt.NDArray]:
    """
    Estimates the vertical shift in pixels from top_stitched_image to bottom_incoming_image.
    Arguments:
        - bottom_incoming_image : shifted image to align
        - top_stitched_image : starting image to align with
        - starting_roi_xyxy: bounding box of the bottom_stitched_image to use for alignment
        - max_shift : maximum vertical shift in pixels
        - horizontal_decimation : subsampling factor in horizontal direction
    Returns:
        pixelshift : Estimated number pixels of that should be added from incoming image. 
        errors : For each pixelshift, the sum of the MAE.
    """
    bottom_incoming_image = bottom_incoming_image[:,::horizontal_decimation,:]
    top_stitched_image = top_stitched_image[:,::horizontal_decimation,:]
    top_stitched_image = bgr2gray(top_stitched_image, dtype=np.uint8)
    bottom_incoming_image  = bgr2gray(bottom_incoming_image, dtype=np.uint8)

    height = bottom_incoming_image.shape[0]
    max_shift = min(max_shift, height)


    roi_x1 = starting_roi_xyxy[0] // horizontal_decimation
    roi_y1 = starting_roi_xyxy[1]
    roi_x2 = starting_roi_xyxy[2] // horizontal_decimation
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
    search = bottom_incoming_image[roi_y1-max_shift:roi_y2:first_stage_decimation, roi_x1:roi_x2]
    template = top_stitched_image[roi_y1:roi_y2:first_stage_decimation, roi_x1:roi_x2]

    res = cv2.matchTemplate(search,template,method)

 
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        # top_left = min_loc
        pixelshift = int(np.argmin(res[::-1, 0]) * first_stage_decimation)
    else:
        # top_left = max_loc
        pixelshift = int(np.argmax(res[::-1, 0]) * first_stage_decimation)

    new_y1 = max(roi_y1 - pixelshift - first_stage_decimation, -bottom_incoming_image.shape[0]) + 1
    new_y2 = min(roi_y2 - pixelshift + first_stage_decimation, roi_y2)
    search = bottom_incoming_image[new_y1:new_y2, roi_x1:roi_x2]
    template = top_stitched_image[roi_y1:roi_y2, roi_x1:roi_x2]

    res = cv2.matchTemplate(search,template,method)


    return pixelshift, res