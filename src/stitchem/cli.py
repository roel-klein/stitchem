import click
import cv2

from pathlib import Path
from functools import partial
from stitchem.align.benchmark import benchmark_alignment_functions
from stitchem.align.match_template import estimate_vertical_shift_match_template
from stitchem.stitch import Stitcher, process_images_with_queue
from stitchem.io import img_generator


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

# @click.command()
@click.group(context_settings=CONTEXT_SETTINGS)
def entrypoint():
    pass


@click.command()
@click.argument("input_directory", type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True))
@click.option("-o", "--output-directory", type=click.Path(file_okay=False, dir_okay=True), default="./output", 
              help="Output directory for stitched images", show_default=True)
@click.option("-t", "--target-height", type=int, default=3500, help="Target height for stitched images.", show_default=True)
@click.option("-s", "--starting-y", type=int, default=100, help="Y coordinate to extract new content from.", show_default=True)
@click.option("-m", "--max-shift", type=int, default=100, help="Maximum expected pixel shift between frames.", show_default=True)
@click.option("--roi-x1", type=int, default=500, help="ROI left x coordinate.", show_default=True)
@click.option("--roi-y1", type=int, default=100, help="ROI top y coordinate.", show_default=True)
@click.option("--roi-x2", type=int, default=3500, help="ROI right x coordinate.", show_default=True)
@click.option("--roi-y2", type=int, default=320, help="ROI bottom y coordinate.", show_default=True)
@click.option("-d", "--horizontal-decimation", type=int, default=16, 
              help="Horizontal decimation factor for preprocessing.")
@click.option("--save/--no-save", default=False, help="Save stitched images to disk.", show_default=True)
@click.option("--visualize/--no-visualize", default=False, help="Show real-time visualization.", show_default=True)
@click.option("--outlier-threshold", type=float, default=2.0, 
              help="Maximum pixels from median to flag as outlier.", show_default=True)
def stitch(input_directory: str, output_directory: str, target_height: int, starting_y: int,
           max_shift: int, roi_x1: int, roi_y1: int, roi_x2: int, roi_y2: int,
           horizontal_decimation: int, save: bool, visualize: bool, outlier_threshold: float):
    """
    Stitch overlapping images from INPUT_DIRECTORY into continuous panoramas.
    
    Images are aligned vertically and merged into target-height outputs.
    Use --visualize to see real-time progress and alignment quality.
    """
    input_dir = Path(input_directory)
    output_dir = Path(output_directory)
    
    click.echo(f"Input directory: {input_dir}")
    click.echo(f"Output directory: {output_dir}")
    click.echo(f"Target height: {target_height}px")
    click.echo(f"ROI: [{roi_x1}, {roi_y1}, {roi_x2}, {roi_y2}]")
    click.echo(f"Horizontal decimation: {horizontal_decimation}x")
    click.echo(f"Visualization: {'enabled' if visualize else 'disabled'}")
    
    # Create stitcher with match template alignment
    starting_roi_xyxy = [roi_x1, roi_y1, roi_x2, roi_y2]
    stitcher = Stitcher(
        partial(estimate_vertical_shift_match_template, 
                max_shift=max_shift, 
                method=cv2.TM_SQDIFF_NORMED),
        horizontal_decimation=horizontal_decimation,
        starting_roi_xyxy=starting_roi_xyxy,
        target_height=target_height,
        starting_y=starting_y,
        max_shift=max_shift,
    )
    
    # Process images
    click.echo("Processing images...")
    gen = img_generator(input_dir)
    completed_count, outliers = process_images_with_queue(
        gen,
        stitcher,
        output_dir,
        save_images=save,
        visualize=visualize,
        outlier_threshold=outlier_threshold,
        window_name=f"{input_dir.name}_stitched"
    )
    
    click.echo(f"\n✓ Completed {completed_count} stitched images")
    
    # Report outliers
    if outliers:
        click.echo(f"\n⚠ Found {len(outliers)} outlier pixelshift values:")
        for frame_idx, shift, reason in outliers[:10]:  # Show first 10
            click.echo(f"  Frame {frame_idx:4d}: shift={shift:5.1f} - {reason}")
        if len(outliers) > 10:
            click.echo(f"  ... and {len(outliers) - 10} more")
    else:
        click.echo("✓ No outliers detected")


@click.command()
@click.argument("input_directory", type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True))
@click.option("-i", "--iterations", default=1,  help="Number of times to repeat aligning all images.", show_default=True)
def benchmark(input_directory : str, iterations : int):
    """
    Benchmark alignment functions on images in given input directory.
    """
    click.echo(f"Starting benchmark on {input_directory}")
    input_directory_path = Path(input_directory)
    # if not input_directory.is_dir():
    #     click.echo(f"Input directory  '{input_directory.__str__}' is not a directory")
    #     return
    benchmark_alignment_functions(input_directory=input_directory_path, iterations=iterations)
    return


entrypoint.add_command(stitch)
entrypoint.add_command(benchmark)

