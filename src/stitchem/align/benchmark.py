import cProfile
import pstats
from pathlib import Path
from functools import partial
from typing import Callable
from time import perf_counter
from collections import defaultdict

import numpy as np
import cv2
from stitchem.io import img_generator
from stitchem.align.bruteforce import estimate_vertical_shift_bruteforce, estimate_vertical_shift_twostage
from stitchem.align.phase_cross_correlation import estimate_shift_phase_cross_correlation_opencv
from stitchem.align.orb import estimate_vertical_shift_orb
from stitchem.align.match_template import estimate_vertical_shift_match_template, estimate_vertical_shift_match_template_twostage

def benchmark_alignment_functions(input_directory : Path,  every_n_frames : int = 1, start_frame : int = 0, iterations=10):
    # preload images
    gen = img_generator(input_directory, every_n_frames=every_n_frames, start=start_frame)
    # starting_image = next(gen)
    # next_image = next(gen)
    images = [i for i in gen]
    print(f"Found {len(images)} images.")

    # NOTE: for phase cross correlation, the roi implicitly defines the max_shift.
    #       This is why it needs a larger roi for the boundary conditions.
    # TODO add ecc https://learnopencv.com/image-alignment-ecc-in-opencv-c-python/ 
    horizontal_decimation=64
    align_funcs: dict[str, Callable[..., object]] = {
        "brute force" : partial(estimate_vertical_shift_bruteforce, starting_roi_xyxy=[500, 100, 3500, 320], max_shift=100, horizontal_decimation=horizontal_decimation),
        "two stage (10th percentile)" : partial(estimate_vertical_shift_twostage, starting_roi_xyxy=[500, 100, 3500, 320], max_shift=100, horizontal_decimation=horizontal_decimation, first_stage_decimation=6, second_stage_percentile=10),
        "two stage (peak neighbours)" : partial(estimate_vertical_shift_twostage, starting_roi_xyxy=[500, 100, 3500, 320], max_shift=100, horizontal_decimation=horizontal_decimation, first_stage_decimation=6, second_stage_percentile=-1),
        # "phase cross correlation (skimage)" : partial(estimate_shift_phase_cross_correlation, starting_roi_xyxy=[500, 0, 3500, 320], horizontal_decimation=horizontal_decimation),
        "phase cross correlation (opencv)" : partial(estimate_shift_phase_cross_correlation_opencv, starting_roi_xyxy=[500, 0, 3500, 320], horizontal_decimation=horizontal_decimation),
        "orb" : partial(estimate_vertical_shift_orb, starting_roi_xyxy=[500, 0, 3500, 320], horizontal_decimation=horizontal_decimation),
        "matchtemplate_twostage" : partial(estimate_vertical_shift_match_template_twostage, starting_roi_xyxy=[500, 100, 3500, 320], max_shift=100, horizontal_decimation=horizontal_decimation, first_stage_decimation=6),
        # "phase cross correlation, upsampled" : partial(estimate_shift_phase_cross_correlation, starting_roi_xyxy=[500, 0, 3500, 320], horizontal_decimation=horizontal_decimation, upsample_factor=10),
    }

    matchtemplate_methods = {
        'TM_CCOEFF' : cv2.TM_CCOEFF, 
        'TM_CCOEFF_NORMED' : cv2.TM_CCOEFF_NORMED,
        # 'TM_CCORR': cv2.TM_CCORR,
        'TM_CCORR_NORMED': cv2.TM_CCORR_NORMED,
        'TM_SQDIFF': cv2.TM_SQDIFF,
        'TM_SQDIFF_NORMED': cv2.TM_SQDIFF_NORMED,
    }
    for (k, v) in matchtemplate_methods.items():
        align_funcs[f"matchtemplate ({k})"] = partial(estimate_vertical_shift_match_template, starting_roi_xyxy=[500, 100, 3500, 320], max_shift=100, horizontal_decimation=horizontal_decimation, method=v)

    average_time = dict()
    total_time = dict()
    estimated_shifts = defaultdict(list)

    for (method, func) in align_funcs.items():
        print("-"*100)
        print(f"Benchmarking {method}")
        # run once before starting
        k1d1 = None
        for i in range(len(images) -1):
            starting_image = images[i]
            next_image = images[i+1]
            if method == "orb":
                result = func(next_image, starting_image, k1d1_precomputed=k1d1)
                k1d1 = result[1]
            else:
                result = func(next_image, starting_image)
            try:

                shift = result[0][0]
                # if shift < 0:
                #     print(shift)
                #     result[0][0] = -result[0][0]
            except:
                shift = result[0]
            estimated_shifts[method].append(shift)

        profiler = cProfile.Profile()
        profiler.enable()

        start = perf_counter()

        for _ in range(iterations):
            k1d1 = None
            for i in range(len(images) -1):
                starting_image = images[i]
                next_image = images[i+1]
                if method == "orb":
                    result = func(next_image, starting_image, k1d1_precomputed=k1d1)
                    k1d1 = result[1]
                else:
                    result = func(next_image, starting_image)
        
        end = perf_counter()
        profiler.disable()
        ps = pstats.Stats(profiler).strip_dirs().sort_stats("cumtime")
        ps.print_stats(20)

        average_time[method] = (end - start) / iterations / (len(images)-1)
        total_time[method] = (end - start)
    
    print("-"*100)
    for method in average_time:
        print(f"{method:<40} average time : {average_time[method]*1000:>8.3f} ms, total time : {total_time[method]:>8.3f} s")
    print("-"*100)
    for method in estimated_shifts:
        shifts = np.array(estimated_shifts[method])
        median = np.median(shifts)
        num_close_enough = ((shifts > (median-1.5)) & (shifts < (median+1.5)))   .sum()/len(shifts) 
        print(f"{method:<40} mean shift: {shifts.mean():8.1f} px, std: {shifts.std():8.3f} px, min: {shifts.min():8.1f} px, max: {shifts.max():8.1f} px, median: {np.median(shifts):8.1f} px, close: {num_close_enough:8.1%} px")

if __name__ == "__main__":
    benchmark_alignment_functions(Path("data/lobster"))








