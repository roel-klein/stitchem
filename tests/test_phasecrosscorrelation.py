from stitchem.align.phase_cross_correlation import estimate_shift_phase_cross_correlation
from stitchem.io import img_generator
from pathlib import Path

import cProfile
import pstats


def test_pcc_shift_estimator():
    gen = img_generator(Path("./data/belt/"))
    img1 = next(gen)
    img2 = next(gen)

    estimate_shift_phase_cross_correlation(
        img2,
        img1,
        starting_roi_xyxy=[1500, 200, 2500, 300],
        horizontal_decimation=4,
    )
    return


if __name__ == "__main__":
    from time import perf_counter
    iterations = 10
    gen = img_generator(Path("./data/belt/"))
    img1 = next(gen)
    img2 = next(gen)


    # profiling
    estimate_shift_phase_cross_correlation(
            img2,
            img1,
            starting_roi_xyxy=[1500, 200, 2500, 300],
            horizontal_decimation=4,
    )

    profiler = cProfile.Profile()
    profiler.enable()

    # profiler.clear()
    start = perf_counter()
    # profiler.enable()
    for i in range(iterations):
        # img2 = img1
        # img1 = next(gen)
        shift_yx, error = estimate_shift_phase_cross_correlation(
            img1,
            img2,
            starting_roi_xyxy=[1500, 200, 2500, 300],
            horizontal_decimation=4,
        )
    end = perf_counter()
    profiler.disable()
    ps = pstats.Stats(profiler).strip_dirs().sort_stats("cumtime")
    ps.print_stats(20)
    print(end - start)
    print(f"{shift_yx=}, {error=}")

