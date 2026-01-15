from typing import Callable
from queue import Queue
from threading import Thread
from pathlib import Path

import numpy as np
import numpy.typing as npt
import cv2

from stitchem.conversion import preprocess

class Stitcher:
    def __init__(
            self, 
            align_func : Callable,
            horizontal_decimation : int,
            starting_roi_xyxy: list[int], 
            target_height : int,
            starting_y : int,   
            max_shift : int,
            blend_zone_height : int = 10,
        ):
        # align parameters
        self.align_func = align_func
        # preprocessing parameters
        # horizontal decimation and starting roi can also be given
        # to most alignment functions, but providing it here allow us
        # to efficiently preprocess each image once, instead of twice
        # (once when previous, once when next image)
        self.horizontal_decimation = horizontal_decimation
        self.starting_roi_xyxy = starting_roi_xyxy

        # align state
        self.previous_img = None
        self.updated_roi_xyxy = None

        # merging parameters
        self.max_shift = max_shift
        self.target_height = target_height
        self.starting_y = starting_y
        self.blend_zone_height = blend_zone_height
        assert self.starting_y >= max_shift, f"starting_y ({self.starting_y}) must be >= max_shift ({max_shift})"
        assert self.blend_zone_height > 0, f"blend_zone_height ({self.blend_zone_height}) must be >= 0"

        # merging state
        self.stitching_progress_img : npt.NDArray | None = None 
        self.stitching_progress_height = 0 
        return
    
    def align(self, bottom_incoming_image) -> int | None:
        bottom_incoming_image, updated_roi_xyxy = preprocess(bottom_incoming_image, self.horizontal_decimation, self.starting_roi_xyxy)
        self.updated_roi_xyxy = updated_roi_xyxy
        if self.previous_img is None:
            self.previous_img = bottom_incoming_image
            return None

        pixelshift_y, _ = self.align_func(bottom_incoming_image, self.previous_img, starting_roi_xyxy=updated_roi_xyxy)
        if pixelshift_y > self.max_shift:
            pixelshift_y = self.max_shift
        self.previous_img = bottom_incoming_image
        return round(pixelshift_y)
    
    def merge(self, img : npt.NDArray, pixelshift_y : int | None) -> npt.NDArray | None:
        if pixelshift_y is None:
            # no previous image, start with first maxshift lines
            pixelshift_y = self.max_shift
        if pixelshift_y <= 0:
            return None
        
        # Ensure we don't try to extract lines before the image starts
        if pixelshift_y > self.starting_y:
            pixelshift_y = self.starting_y
        
        # initialize the stitched image shape
        if self.stitching_progress_img is None:
            stitched_shape = (self.target_height, img.shape[1], img.shape[2]) if img.ndim == 3 else (self.target_height, img.shape[1])
            self.stitching_progress_img = np.zeros(stitched_shape, dtype=img.dtype)
            self.stitching_progress_height = 0
        assert self.stitching_progress_img is not None
        # Extract new lines once
        end_idx = self.stitching_progress_height + pixelshift_y
        
        completed_image = None
        if self.stitching_progress_height == 0:
            # First batch - direct assignment
            self.stitching_progress_img[:pixelshift_y] = img[self.starting_y-pixelshift_y:self.starting_y]
            self.stitching_progress_height = pixelshift_y
        elif end_idx <= self.target_height:
            # Append new content at the current height
            self.stitching_progress_img[self.stitching_progress_height:end_idx] = img[self.starting_y-pixelshift_y:self.starting_y]
            self.stitching_progress_height = end_idx
        else:
            # Need to complete current image and start next one
            remaining_space = self.target_height - self.stitching_progress_height
            split_point = self.starting_y - pixelshift_y + remaining_space
            
            # Fill remaining space
            self.stitching_progress_img[self.stitching_progress_height:self.target_height] = img[self.starting_y-pixelshift_y:split_point]
            completed_image = self.stitching_progress_img
            
            # Start new image with remaining lines
            opening_lines_count = pixelshift_y - remaining_space
            self.stitching_progress_img = np.zeros_like(self.stitching_progress_img)
            self.stitching_progress_img[:opening_lines_count] = img[split_point:self.starting_y]
            self.stitching_progress_height = opening_lines_count
        return completed_image
    
    def merge_linear_blend(self, img : npt.NDArray, pixelshift_y : int ) -> npt.NDArray | None:
        """Merge with linear blending in the overlap region for smoother transitions."""
        if pixelshift_y is None:
            # no previous image, start with first maxshift lines
            pixelshift_y = self.max_shift
        if pixelshift_y <= 0:
            return None
        
        # Ensure we don't try to extract lines before the image starts
        if pixelshift_y > self.starting_y:
            pixelshift_y = self.starting_y
        
        # initialize the stitched image shape
        if self.stitching_progress_img is None:
            stitched_shape = (self.target_height, img.shape[1], img.shape[2]) if img.ndim == 3 else (self.target_height, img.shape[1])
            self.stitching_progress_img = np.zeros(stitched_shape, dtype=img.dtype)
            self.stitching_progress_height = 0

        # Determine blend zone (can't be larger than available space)
        effective_blend_height = min(self.blend_zone_height, self.stitching_progress_height, pixelshift_y)
        
        # Calculate extraction indices
        # Extract pixelshift + blend zone, where blend zone overlaps with existing content
        extract_start = self.starting_y - pixelshift_y - effective_blend_height
        extract_end = self.starting_y
        
        # Extract new lines plus blend zone
        end_idx = self.stitching_progress_height + pixelshift_y
        
        completed_image = None
        if self.stitching_progress_height == 0:
            # First batch - direct assignment (no blending possible)
            self.stitching_progress_img[:pixelshift_y] = img[self.starting_y-pixelshift_y:self.starting_y]
            self.stitching_progress_height = pixelshift_y
        elif end_idx <= self.target_height:
            # Append new content at the current height
            if effective_blend_height > 0:
                # Apply linear blending in the overlap region
                blend_start_idx = self.stitching_progress_height - effective_blend_height
                extracted = img[extract_start:extract_end]
                
                # Create linear alpha blend weights
                alpha = np.linspace(0, 1, effective_blend_height, dtype=np.float32)
                if img.ndim == 3:
                    alpha = alpha[:, np.newaxis, np.newaxis]  # Shape for broadcasting with color images
                else:
                    alpha = alpha[:, np.newaxis]  # Shape for broadcasting with grayscale
                
                # Blend the overlap region
                existing = self.stitching_progress_img[blend_start_idx:self.stitching_progress_height].astype(np.float32)
                new_overlap = extracted[:effective_blend_height].astype(np.float32)
                blended = (1 - alpha) * existing + alpha * new_overlap
                
                # Write blended overlap and new content
                self.stitching_progress_img[blend_start_idx:self.stitching_progress_height] = blended.astype(img.dtype)
                self.stitching_progress_img[self.stitching_progress_height:end_idx] = extracted[effective_blend_height:]
            else:
                # No blending - direct assignment
                self.stitching_progress_img[self.stitching_progress_height:end_idx] = img[self.starting_y-pixelshift_y:self.starting_y]
            
            self.stitching_progress_height = end_idx
        else:
            # Need to complete current image and start next one
            remaining_space = self.target_height - self.stitching_progress_height
            
            if effective_blend_height > 0:
                # Complex case: blending + split across two images
                blend_start_idx = self.stitching_progress_height - effective_blend_height
                extracted = img[extract_start:extract_end]
                
                # Create linear alpha blend weights
                alpha = np.linspace(0, 1, effective_blend_height, dtype=np.float32)
                if img.ndim == 3:
                    alpha = alpha[:, np.newaxis, np.newaxis]
                else:
                    alpha = alpha[:, np.newaxis]
                
                # Blend the overlap region
                existing = self.stitching_progress_img[blend_start_idx:self.stitching_progress_height].astype(np.float32)
                new_overlap = extracted[:effective_blend_height].astype(np.float32)
                blended = (1 - alpha) * existing + alpha * new_overlap
                self.stitching_progress_img[blend_start_idx:self.stitching_progress_height] = blended.astype(img.dtype)
                
                # Now handle the new content that needs to be split
                new_content = extracted[effective_blend_height:]
                new_content_height = len(new_content)
                
                if new_content_height <= remaining_space:
                    # All new content fits in current image
                    self.stitching_progress_img[self.stitching_progress_height:self.stitching_progress_height + new_content_height] = new_content
                    self.stitching_progress_height += new_content_height
                else:
                    # Split new content across images
                    self.stitching_progress_img[self.stitching_progress_height:self.target_height] = new_content[:remaining_space]
                    completed_image = self.stitching_progress_img
                    
                    # Start new image with remaining content
                    opening_lines_count = new_content_height - remaining_space
                    self.stitching_progress_img = np.zeros_like(self.stitching_progress_img)
                    self.stitching_progress_img[:opening_lines_count] = new_content[remaining_space:]
                    self.stitching_progress_height = opening_lines_count
            else:
                # No blending - use original logic
                split_point = self.starting_y - pixelshift_y + remaining_space
                
                # Fill remaining space
                self.stitching_progress_img[self.stitching_progress_height:self.target_height] = img[self.starting_y-pixelshift_y:split_point]
                completed_image = self.stitching_progress_img
                
                # Start new image with remaining lines
                opening_lines_count = pixelshift_y - remaining_space
                self.stitching_progress_img = np.zeros_like(self.stitching_progress_img)
                self.stitching_progress_img[:opening_lines_count] = img[split_point:self.starting_y]
                self.stitching_progress_height = opening_lines_count
        
        return completed_image
    
    def flush(self) -> npt.NDArray | None:
        if self.stitching_progress_img is not None:
            incomplete_img = self.stitching_progress_img.copy()
        else:
            incomplete_img = None
        self.stitching_progress_img = None
        self.previous_img = None
        return incomplete_img


def process_images_with_queue(
    image_generator,
    stitcher: Stitcher,
    output_dir,
    save_images: bool = True,
    visualize: bool = False,
    load_queue_size: int = 10,
    write_queue_size: int = 5,
    window_name: str = "stitched_image",
    outlier_threshold: float = 2.0,
):
    """
    Process images from a generator using queued I/O for efficiency.
    
    Args:
        image_generator: Generator yielding numpy array images
        stitcher: Configured Stitcher instance
        output_dir: Path to save stitched images
        save_images: Whether to actually write images to disk
        visualize: Whether to display images in a GUI window
        load_queue_size: Maximum images to buffer during loading
        write_queue_size: Maximum images to buffer during writing
        window_name: Name of the visualization window
        outlier_threshold: Maximum pixels from median to consider outlier
    
    Returns:
        Tuple of (completed_count, outlier_list)
    """

    
    output_dir = Path(output_dir)
    
    def image_loader(gen, output_queue):
        """Load images from generator and put them in a queue."""
        for idx, img in enumerate(gen):
            output_queue.put((idx, img))
        output_queue.put(None)  # Sentinel to signal completion
    
    def image_writer(input_queue):
        """Write images in a separate thread from a queue."""
        while True:
            item = input_queue.get()
            if item is None:  # Sentinel to signal completion
                input_queue.task_done()
                break
            path, img = item
            path.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(path), img)
            print(f"  Saved: {path}")
            input_queue.task_done()
    
    # Setup queues
    load_queue: Queue = Queue(maxsize=load_queue_size)
    write_queue: Queue = Queue(maxsize=write_queue_size)
    
    # Start loader thread
    loader_thread = Thread(target=image_loader, args=(image_generator, load_queue), daemon=True)
    loader_thread.start()
    
    # Start writer thread if saving
    if save_images:
        writer_thread = Thread(target=image_writer, args=(write_queue,), daemon=True)
        writer_thread.start()
    
    # Setup visualization window if needed
    paused = False
    if visualize:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.namedWindow("Preprocessed Images", cv2.WINDOW_NORMAL)
        # Resize windows to larger size (adjust as needed)
        cv2.resizeWindow(window_name, 1600, 1200)
        cv2.resizeWindow("Preprocessed Images", 1600, 800)
        print("\nVisualization controls:")
        print("  SPACE - Pause/Resume")
        print("  ENTER - Step forward one frame (when paused)")
        print("  's'   - Save current progress image")
        print("  'q'   - Quit")
    
    # Process images
    completed_count = 0
    pixelshift_history = []
    outliers = []  # List of (frame_idx, pixelshift, reason) tuples
    while True:
        item = load_queue.get()
        if item is None:  # End of images
            load_queue.task_done()
            break
        
        idx, img = item
        
        # Align with previous image
        if visualize:
            if stitcher.previous_img is not None:
                prep_previous_vis = stitcher.previous_img.copy()
            else:
                prep_previous_vis = None 
        pixelshift = stitcher.align(img)
        
        # Track pixelshift and detect outliers
        is_outlier = False
        if pixelshift is not None:
            pixelshift_history.append(pixelshift)
            
            # Detect outliers (after we have enough data)
            if len(pixelshift_history) >= 10:
                median_shift = np.median(pixelshift_history[-20:])  # Use recent history
                
                # Check if pixelshift is more than outlier_threshold pixels from median
                deviation = abs(pixelshift - median_shift)
                if deviation > outlier_threshold:
                    is_outlier = True
                    reason = f"Deviation: {deviation:.1f} pixels from median {median_shift:.1f}"
                    outliers.append((idx, pixelshift, reason))
                    print(f"  ⚠ Outlier detected at frame {idx}: shift={pixelshift}, {reason}")
                    if visualize:
                        paused = True
                        print("  ⏸ Auto-paused on outlier")
                
                # Also flag if shift equals max_shift (capped)
                if pixelshift == stitcher.max_shift:
                    is_outlier = True
                    reason = f"Capped at max_shift ({stitcher.max_shift})"
                    if (idx, pixelshift, reason) not in outliers:
                        outliers.append((idx, pixelshift, reason))
                        print(f"  ⚠ Outlier detected at frame {idx}: shift={pixelshift}, {reason}")
                        if visualize:
                            paused = True
                            print("  ⏸ Auto-paused on outlier")
        
        # Choose visualization color based on outlier status
        viz_color = (0, 0, 255) if is_outlier else (0, 255, 0)  # Red if outlier, green otherwise
        
        # Merge and get completed image if available
        completed_img = stitcher.merge(img, pixelshift)
        
        # Visualize the preprocessed images and alignment info
        if visualize and prep_previous_vis is not None:
            # Create visualization for preprocessed images
            prep_current_vis = stitcher.previous_img.copy()
            prep_current_vis = cv2.cvtColor(prep_current_vis, cv2.COLOR_GRAY2BGR)
            prep_previous_vis = cv2.cvtColor(prep_previous_vis, cv2.COLOR_GRAY2BGR)
            # Mirror current image horizontally for easier comparison
            prep_current_vis = cv2.flip(prep_current_vis, 1)

            # Calculate padding needed to align images based on pixelshift
            # Pixelshift represents how much NEW content is in current vs previous
            # Positive shift means camera moved up, so current frame is HIGHER in the scene
            if pixelshift is not None and pixelshift > 0:
                # Current frame is higher - pad it at TOP to align with previous
                pad_previous_top = 0
                pad_previous_bottom = pixelshift
                pad_current_top = pixelshift
                pad_current_bottom = 0
            elif pixelshift is not None and pixelshift < 0:
                # Current frame is lower - pad it at BOTTOM
                pad_previous_top = abs(pixelshift)
                pad_previous_bottom = 0
                pad_current_top = 0
                pad_current_bottom = abs(pixelshift)
            else:
                # No shift or first frame
                pad_previous_top = pad_previous_bottom = 0
                pad_current_top = pad_current_bottom = 0
            
            # Apply padding to align images
            prep_previous_vis = cv2.copyMakeBorder(
                prep_previous_vis, pad_previous_top, pad_previous_bottom, 0, 0,
                cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
            prep_current_vis = cv2.copyMakeBorder(
                prep_current_vis, pad_current_top, pad_current_bottom, 0, 0,
                cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
            
            # Draw grid lines for easier alignment comparison
            grid_spacing = 50  # pixels between grid lines
            grid_color = (80, 80, 80)  # dark gray
            
            # Horizontal grid lines
            for y in range(0, prep_previous_vis.shape[0], grid_spacing):
                cv2.line(prep_previous_vis, (0, y), (prep_previous_vis.shape[1], y), grid_color, 1)
                cv2.line(prep_current_vis, (0, y), (prep_current_vis.shape[1], y), grid_color, 1)
            
            # Vertical grid lines
            for x in range(0, prep_previous_vis.shape[1], grid_spacing):
                cv2.line(prep_previous_vis, (x, 0), (x, prep_previous_vis.shape[0]), grid_color, 1)
                cv2.line(prep_current_vis, (x, 0), (x, prep_current_vis.shape[0]), grid_color, 1)

            # Draw ROI on both images (adjust for padding)
            if stitcher.updated_roi_xyxy is not None:
                x1, y1, x2, y2 = stitcher.updated_roi_xyxy
                cv2.rectangle(prep_previous_vis, (x1, y1 + pad_previous_top), (x2, y2 + pad_previous_top), (255, 0, 0), 2)
                # Current image is flipped horizontally, so mirror the x-coordinates
                width = prep_current_vis.shape[1]
                x1_flipped = width - x2
                x2_flipped = width - x1
                cv2.rectangle(prep_current_vis, (x1_flipped, y1 - stitcher.max_shift + pad_current_top), (x2_flipped, y2 + pad_current_top), (255, 0, 0), 2)
                
                # Draw pixelshift visualization if we have one
                if pixelshift is not None:
                    # Draw the shifted ROI on current image to show alignment
                    shifted_y1 = y1 - pixelshift + pad_current_top
                    shifted_y2 = y2 - pixelshift + pad_current_top
                    cv2.rectangle(prep_current_vis, (x1_flipped, shifted_y1), (x2_flipped, shifted_y2), viz_color, 2)
                    
                    # Draw arrow showing the shift on current image (mirrored position)
                    arrow_x = x1_flipped - 10
                    arrow_start_y = y1 - stitcher.max_shift + 20 + pad_current_top
                    arrow_end_y = shifted_y1 + 20
                    cv2.arrowedLine(prep_current_vis, (arrow_x, arrow_start_y), (arrow_x, arrow_end_y), 
                                   viz_color, 3, tipLength=0.3)
                    
                    # Add pixelshift text (mirrored position)
                    text_x = max(0, arrow_x - 100)  # Position to the left of arrow
                    cv2.putText(prep_current_vis, f"shift={pixelshift}", 
                               (text_x, (arrow_start_y + arrow_end_y) // 2),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, viz_color, 2)

            prep_vis = np.hstack((prep_previous_vis, prep_current_vis))
            cv2.imshow("Preprocessed Images", prep_vis)
        
        # Visualize the in-progress stitching
        if visualize and stitcher.stitching_progress_img is not None:
            # Show the progress image with a marker line for current height
            progress_vis = stitcher.stitching_progress_img.copy()
            if stitcher.stitching_progress_height > 0:
                # Draw a line showing current fill height
                cv2.line(progress_vis, 
                        (0, stitcher.stitching_progress_height), 
                        (progress_vis.shape[1], stitcher.stitching_progress_height),
                        viz_color, 2)
                # Add text overlay showing status
                status_text = f"Frame: {idx} | Height: {stitcher.stitching_progress_height}/{stitcher.target_height} | Pixelshift: {pixelshift if pixelshift is not None else 'N/A'}"
                if is_outlier:
                    status_text += " | OUTLIER"
                if paused:
                    status_text += " | PAUSED"
                cv2.putText(progress_vis, status_text, (10, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 4, viz_color, 4)
            
            cv2.imshow(window_name, progress_vis)
            
            # Handle pause/step logic
            wait_time = 0 if paused else 1
            while True:
                key = cv2.waitKey(wait_time)
                
                if key == ord('q') or key == 27:  # q or ESC to quit
                    print("Visualization stopped by user")
                    load_queue.task_done()
                    # Drain the queue
                    while True:
                        try:
                            item = load_queue.get_nowait()
                            if item is None:
                                break
                            load_queue.task_done()
                        except:
                            break
                    return completed_count
                    
                elif key == ord(' '):  # SPACE to toggle pause
                    paused = not paused
                    print(f"{'Paused' if paused else 'Resumed'}")
                    if not paused:
                        break  # Exit wait loop and continue processing
                    wait_time = 0  # Continue waiting for next key
                    
                elif key == 13:  # ENTER to step when paused
                    if paused:
                        print(f"Step: Processing frame {idx}")
                        break  # Process one frame then pause again
                    
                elif key == ord('s'):  # s to save current progress
                    save_path = output_dir / f"progress_{idx:04d}.png"
                    cv2.imwrite(str(save_path), progress_vis)
                    print(f"  Manually saved progress: {save_path}")
                    if paused:
                        wait_time = 0  # Continue waiting
                    else:
                        break
                        
                elif key == -1:  # No key pressed
                    if not paused:
                        break  # Continue normal processing
                    wait_time = 0  # Keep waiting for input when paused
                else:
                    break  # Unknown key, continue
        
        if completed_img is not None:
            output_path = output_dir / f"{window_name}_{completed_count:04d}.png"
            
            if save_images:
                write_queue.put((output_path, completed_img))
            elif not visualize:
                print(f"  Would save: {output_path}")
            
            completed_count += 1
        
        load_queue.task_done()
    
    # Save any remaining incomplete image
    incomplete_img = stitcher.flush()
    if incomplete_img is not None:
        output_path = output_dir / f"{window_name}_{completed_count:04d}_remainder.png"
        
        if visualize:
            cv2.imshow(window_name, incomplete_img)
            print("Press any key to close visualization...")
            cv2.waitKey(0)
        
        if save_images:
            write_queue.put((output_path, incomplete_img))
        elif not visualize:
            print(f"  Would save incomplete: {output_path}")
    
    # Cleanup visualization
    if visualize:
        cv2.destroyAllWindows()
    
    # Signal writer to stop and wait for completion
    loader_thread.join()
    if save_images:
        write_queue.put(None)
        write_queue.join()
        writer_thread.join()
    
    return completed_count, outliers


if __name__ == "__main__":
    from pathlib import Path
    from functools import partial
    from stitchem.align.match_template import estimate_vertical_shift_match_template
    from stitchem.io import img_generator
    from time import perf_counter
    import cProfile
    import pstats
    
    indir = Path("../FDF/ids_calibration_data/moving_fish5/")
    outdir = Path("./output")
    
    s = Stitcher(
        partial(estimate_vertical_shift_match_template, max_shift=100),
        horizontal_decimation=4,
        starting_roi_xyxy=[500, 100, 3500, 320],
        target_height=3500,
        starting_y=100,
        max_shift=100,
        blend_zone_height=20,
    )

    # Warm up JIT compilation
    bgr2gray(np.zeros((100, 100, 3), dtype=np.uint8), dtype=np.uint8)
    
    profiler = cProfile.Profile()
    profiler.enable()
    start = perf_counter()

    # Process images using the queue-based function
    gen = img_generator(indir)
    completed_count, outliers = process_images_with_queue(
        gen, 
        s, 
        outdir,
        save_images=True,  # Set to True to actually write images 
        visualize=True,     # Set to True to display in GUI (press 'q' to quit, 's' to save)
        outlier_threshold=2.0,  # Standard deviations from median to flag as outlier
    )
    
    print(f"\nTotal completed images: {completed_count}")
    
    # Print outlier summary
    if outliers:
        print(f"\n⚠ Found {len(outliers)} outlier pixelshift values:")
        for frame_idx, shift, reason in outliers:
            print(f"  Frame {frame_idx:4d}: shift={shift:5.1f} - {reason}")
    else:
        print("\n✓ No outliers detected")
    
    end = perf_counter()
    profiler.disable()
    ps = pstats.Stats(profiler).strip_dirs().sort_stats("cumtime")
    ps.print_stats(20)
    print(f"total time : {(end-start)*1:>8.3f} s")
