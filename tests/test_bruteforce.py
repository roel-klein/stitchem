from stitchem.align.bruteforce import estimate_vertical_shift_bruteforce, estimate_vertical_shift_twostage
from stitchem.io import img_generator
from pathlib import Path

import cProfile
import pstats




def test_bruteforce_shift_estimator():
    gen = img_generator(Path("./data/belt/"))
    img1 = next(gen)
    img2 = next(gen)


    estimate_vertical_shift_bruteforce(
        img1,
        img2,
        [1500, 200, 2500, 300],
        max_shift=200,
        horizontal_decimation=4,
    )

    return

def test_twostage_shift_estimator():
    gen = img_generator(Path("./data/belt/"))
    img1 = next(gen)
    img2 = next(gen)

    estimate_vertical_shift_twostage(
        img1,
        img2,
        [1500, 200, 2500, 300],
        max_shift=200,
        horizontal_decimation=4,
        first_stage_decimation=4,
    )


if __name__ == "__main__":
    from time import perf_counter
    iterations = 10
    gen = img_generator(Path("./data/belt/"))
    img1 = next(gen)
    img2 = next(gen)


    # profiling

    estimate_vertical_shift_bruteforce(
            img1,
            img2,
            [1500, 200, 2500, 300],
            max_shift=200,
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
        estimate_vertical_shift_bruteforce(
            img2,
            img1,
            [1500, 200, 2500, 300],
            max_shift=200,
            horizontal_decimation=4,
        )
    end = perf_counter()
    profiler.disable()
    ps = pstats.Stats(profiler).strip_dirs().sort_stats("cumtime")
    ps.print_stats(20)
    print(end - start)
    
    profiler.clear()

    estimate_vertical_shift_twostage(
            img2,
            img1,
            [1500, 200, 2500, 300],
            max_shift=200,
            horizontal_decimation=1,
            first_stage_decimation=8,
    )
    # profiler.enable()

    start = perf_counter()
    # profiler.enable()
    for i in range(iterations):
        # img2 = img1
        # img1 = next(gen)
        pixelshift, errors = estimate_vertical_shift_twostage(
            img2,
            img1,
            [1500, 200, 2500, 300],
            max_shift=200,
            horizontal_decimation=4,
            first_stage_decimation=8,
        )
    end = perf_counter()
    # profiler.disable()
    # ps = pstats.Stats(profiler).strip_dirs().sort_stats("cumtime")
    # ps.print_stats(20)
    print(end - start)
    print(f"{pixelshift=}")

