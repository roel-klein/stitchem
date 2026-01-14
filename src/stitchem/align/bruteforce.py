import warnings
import  numpy.typing as npt
import numpy as np
import cv2

from stitchem.conversion import bgr2gray

def estimate_vertical_shift_bruteforce(
        bottom_incoming_image : npt.NDArray, 
        top_stitched_image: npt.NDArray, 
        starting_roi_xyxy: list[int], 
        max_shift : int,
        horizontal_decimation : int = 1,
    ) -> tuple[int, npt.NDArray[np.float64]]:
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
    top_stitched_image = bgr2gray(top_stitched_image, dtype=np.float32)
    bottom_incoming_image  = bgr2gray(bottom_incoming_image, dtype=np.float32)
    # top_incoming_image = cv2.cvtColo=()

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

    top_stitched_image_roi = top_stitched_image[roi_y1:roi_y2,roi_x1:roi_x2]
    errors = np.empty(max_shift)
    for i in range(max_shift):
        bottom_incoming_image_roi = bottom_incoming_image[roi_y1-i:roi_y2-i,roi_x1:roi_x2]
        diff = cv2.absdiff(top_stitched_image_roi, bottom_incoming_image_roi)
        errors[i] = diff.sum()
    pixel_offset = int(np.argmin(errors))
    return pixel_offset, errors

def estimate_vertical_shift_twostage(
        bottom_incoming_image : npt.NDArray, 
        top_stitched_image: npt.NDArray, 
        starting_roi_xyxy: list[int], 
        max_shift : int,
        horizontal_decimation : int = 1,
        first_stage_decimation : int = 2,
        second_stage_decimation : int = 1,
        second_stage_percentile : float = 10,
    ) -> tuple[int, npt.NDArray[np.float64]]:
    
    
    """
    Estimates the vertical shift in pixels from top_stitched_image to bottom_incoming_image.
    Arguments:
        - bottom_incoming_image : shifted image to align
        - top_stitched_image : starting image to align with
        - starting_roi_xyxy: bounding box of the bottom_stitched_image to use for alignment
        - max_shift : maximum vertical shift in pixels
        - horizontal_decimation : subsampling factor in horizontal direction
        - first_stage_decimation : step size for shifts to consider in the first stage.
                after which the top ranges are selected to be checked in the second stage
                default=2.
        - second_stage_decimation : step size for shifts to consider in the second/final stage.
                default=1.
        - second_stage_percentile : Top-n percentile of first-stage shifts to interpolate in the second/final stage.
                default=1.
        
    Returns:
        pixelshift : Estimated number pixels of that should be added from incoming image. 
        errors : For each pixelshift, the sum of the MAE. For skipped pixelshift steps, np.nan. 
    """
    bottom_incoming_image = bottom_incoming_image[:,::horizontal_decimation,:]
    top_stitched_image = top_stitched_image[:,::horizontal_decimation,:]
    
    # for some reason the bgr2gray function is faster here
    # than using cv2.cvtColor or simply taking one of the channels
    # this seems strange ... 
    # using the functions standalone, it is not...
    top_stitched_image = bgr2gray(top_stitched_image, dtype=np.float32)
    bottom_incoming_image  = bgr2gray(bottom_incoming_image, dtype=np.float32)
    # bottom_stitched_image = cv2.cvtColor(bottom_stitched_image, cv2.COLOR_BGR2GRAY)
    # top_incoming_image = cv2.cvtColor(top_incoming_image, cv2.COLOR_BGR2GRAY)
    # pseudo-bgr2gray by taking g-channel
    # bottom_stitched_image = bottom_stitched_image[:, :, 1] 
    # top_incoming_image  = top_incoming_image[:, :, 1]
    # bgr2gray function is a njit-ed 
    # bottom_stitched_image = 0.299 * bottom_stitched_image[:,:, 2] + 0.587 * bottom_stitched_image[:,:, 1] + 0.114 * bottom_stitched_image[:,:, 0]
    # top_incoming_image = 0.299 * top_incoming_image[:,:, 2] + 0.587 * top_incoming_image[:,:, 1] + 0.114 * top_incoming_image[:,:, 0]
    
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

    top_stitched_image_roi = top_stitched_image[roi_y1:roi_y2,roi_x1:roi_x2]
    
    # stage 1 
    indices = np.arange(max_shift)
    shifts_firststage = indices[::first_stage_decimation]
    errors_firststage = np.empty(len(shifts_firststage))
    for i, j in enumerate(shifts_firststage):
        bottom_incoming_image_roi = bottom_incoming_image[roi_y1-j:roi_y2-j,roi_x1:roi_x2]
        diff = cv2.absdiff(bottom_incoming_image_roi, top_stitched_image_roi)
        errors_firststage[i] = diff.sum()

    # find shifts for stage 2
    if second_stage_percentile < 0:
        first_stage_bestshift = int(np.argmin(errors_firststage) * first_stage_decimation)

        start =  int(max(first_stage_bestshift - first_stage_decimation + 1, 0))
        end =  int(min(first_stage_bestshift + first_stage_decimation, max_shift))
        # end = min(roi_y2 - first_stage_bestshift + first_stage_decimation, roi_y2)
        shifts_secondstage = {
             i for i in range(start, end)
        }
        shifts_secondstage.remove(first_stage_bestshift)
    else:
        error_threshold = np.percentile(errors_firststage, second_stage_percentile)
        shifts_secondstage = set()
        for i, e in zip(shifts_firststage, errors_firststage):
            if e > error_threshold:
                continue
            for j in range(second_stage_decimation, first_stage_decimation, second_stage_decimation):
                posj = i + j
                negj = i - j
                if posj < max_shift:
                    shifts_secondstage.add(posj)
                if negj > 0:
                    shifts_secondstage.add(negj)
            
    errors_secondstage = dict()

    # stage 2
    for i in shifts_secondstage:
        bottom_incoming_image_roi = bottom_incoming_image[roi_y1-i:roi_y2-i,roi_x1:roi_x2]
        diff = cv2.absdiff(bottom_incoming_image_roi, top_stitched_image_roi)
        errors_secondstage[i] = diff.sum()
    
    errors = []
    for i in range(max_shift):
        if i % first_stage_decimation == 0:
            errors.append(errors_firststage[i // first_stage_decimation])
        else:
            errors.append(errors_secondstage.get(i, np.nan))
    errors_array = np.array(errors)
    
    pixel_offset = int(np.nanargmin(errors))
    return pixel_offset, errors_array
    