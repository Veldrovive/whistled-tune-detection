import matplotlib.pyplot as plt
import numpy as np
from audio_lib_v2 import AudioStreamProcessor
from scipy.ndimage import convolve, label
import skimage as ski
from pathlib import Path
import traceback

data_path = Path(__file__).parent / "data"

BUFFER_DURATION = 5
N_MELS = 256
REFRESH_RATE_HZ = 10
FMIN = 300
FMAX = 3500
CENTROID_DETECTION_WIDTH = 5

def create_horizontal_line_filter(line_height: int, filter_width: int = 3) -> np.ndarray:
    """
    Constructs a convolution filter (kernel) for detecting horizontal lines
    of a specific height.

    The filter is designed to be insensitive to the absolute magnitude of the
    surrounding area due to its zero-sum property. It works by creating an
    "excitatory" central region and "inhibitory" outer regions.

    The weights are chosen such that the sum of all elements is zero.
    Based on the rule: N * A = 2 * B, where N is the line height,
    A is the positive weight, and B is the magnitude of the negative weight.
    A simple integer solution is A=2 and B=N.

    Args:
        line_height (int): The height in pixels of the horizontal line to detect (N).
                           Must be a positive integer.
        filter_width (int, optional): The width of the filter. A smaller width detects
                                      sharper lines. Defaults to 3.

    Returns:
        np.ndarray: A 2D NumPy array representing the convolution kernel.
                    The shape will be (line_height + 2, filter_width).
    
    Raises:
        ValueError: If line_height or filter_width are not positive integers.
    """
    if not isinstance(line_height, int) or line_height < 1:
        raise ValueError("line_height must be a positive integer.")
    if not isinstance(filter_width, int) or filter_width < 1:
        raise ValueError("filter_width must be a positive integer.")

    # Determine the weights to ensure the filter sums to zero.
    # We use the simple A=2, B=N solution.
    positive_weight = 2
    negative_weight = -line_height

    # Create the three components of the filter
    # 1. Top inhibitory row
    top_row = np.full(shape=(1, filter_width), fill_value=negative_weight)
    
    # 2. Central excitatory block for the line itself
    middle_block = np.full(shape=(line_height, filter_width), fill_value=positive_weight)
    
    # 3. Bottom inhibitory row
    bottom_row = np.full(shape=(1, filter_width), fill_value=negative_weight)

    # Stack the components vertically to form the final kernel
    kernel = np.vstack([top_row, middle_block, bottom_row])
    
    # The sum should be exactly zero by construction, but an assertion is good practice
    assert np.sum(kernel) == 0, "Internal error: Kernel sum is not zero."

    # Return as a float type, which is standard for convolution operations
    return kernel.astype(np.float32) / np.sum(np.abs(kernel))

def main():
    processor = AudioStreamProcessor(
        chunk_size=1024,
        rate=22050,
        n_fft=2048,
        hop_len=512,
        buffer_duration_s=BUFFER_DURATION,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        wav_filepath=None, # data_path / "whistle_test.m4a",
        device_name="MacBook Pro Microphone"
    )

    plt.ion()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    initial_spectrogram = np.zeros((N_MELS, processor.num_buffer_samples))

    # --- Setup for the top plot (Raw Spectrogram) ---
    im1 = ax1.imshow(initial_spectrogram, aspect='auto', origin='lower', cmap='viridis')
    fig.colorbar(im1, ax=ax1, format='%+2.0f dB')
    ax1.set_title("Real-time Raw Spectrogram")
    ax1.set_ylabel("Frequency (Hz)")

    # --- Setup for plot 2 (Processed Spectrogram) ---
    im2 = ax2.imshow(initial_spectrogram, aspect='auto', origin='lower', cmap='inferno')
    fig.colorbar(im2, ax=ax2)
    ax2.set_title("Processed Spectrogram (Detected Blobs)")
    ax2.set_xlabel("Time (chunks)")
    ax2.set_ylabel("Frequency (Hz)")

    # --- Setup for the bottom plot (Centroids) ---
    # We initialize an empty line plot with red circle markers. We'll update its data later.
    centroid_plot_line, = ax3.plot([], [], 'ro', markersize=8, alpha=0.7)
    ax3.set_title("Detected Centroids")
    ax3.set_xlabel("Time (chunks)")
    ax3.set_ylabel("Frequency (Hz)")
    # Set the plot limits to match the spectrogram axes
    ax3.set_xlim(0, processor.num_buffer_samples)
    ax3.set_ylim(0, N_MELS - 1)
    ax3.grid(True, linestyle=':', alpha=0.6)

    # --- Setup frequency ticks for both plots ---
    freqs = processor.mel_center_freqs
    tick_indices = np.linspace(0, len(freqs) - 1, num=10, dtype=int)
    for ax in [ax1, ax2]:
        ax.set_yticks(tick_indices)
        ax.set_yticklabels([f"{freqs[i]:.0f}" for i in tick_indices])

    try:
        tracked_curves = []
        for spec_buffer, _ in processor.listen():
            # print(processor.peaks_buffer[-1])
            if len(spec_buffer) > 0:
                # 1. Get raw dB spectrogram data
                spec_array_db = np.array(spec_buffer).T
                num_current_frames = spec_array_db.shape[1]

                peaks_raw = np.full((N_MELS, len(processor.peaks_buffer)), 0, dtype=np.float32)
                for i, (peaks, properties) in enumerate(processor.peaks_buffer):
                    peaks_raw[peaks, i] = properties["prominences"]
                peaks_raw = np.where(peaks_raw > 30, peaks_raw, 0)
                # peaks_full = np.full((N_MELS, processor.num_buffer_samples), 0, dtype=np.float32)
                # peaks_full[:, -num_current_frames:] = peaks_raw

                # --- Update RAW spectrogram plot (im1) ---
                full_spec_raw = np.full((N_MELS, processor.num_buffer_samples), -80.0) # Fill with silence
                full_spec_raw[:, -num_current_frames:] = spec_array_db
                im1.set_data(full_spec_raw)
                im1.set_clim(vmin=-60, vmax=np.max(full_spec_raw)) # Use a fixed vmin for stability

                # --- Process the spectrogram ---
                gaussian_kernel = np.array([[1, 2, 1],
                                            [2, 4, 2],
                                            [1, 2, 1]]) / 16.0
                spec_array_smooth = convolve(spec_array_db, gaussian_kernel, mode='reflect')

                # y_gradient_kernel = np.array([
                #     [-1, -1, -1],
                #     [2, 2, 2],
                #     [-1, -1, -1]
                # ])
                y_gradient_kernel = create_horizontal_line_filter(6, 5)
                spec_array_y_grad = convolve(spec_array_smooth, y_gradient_kernel, mode='reflect')

                # TODO: Implement a system that looks for both large magnitude in dB and in the y grad metric

                # spec_array_binary = spec_array_smooth > 0
                spec_array_binary = np.logical_and(spec_array_y_grad > 1, spec_array_smooth > -15)

                # footprint = ski.morphology.disk(12)
                footprint_close = ski.morphology.rectangle(nrows=3, ncols=7)
                spec_array_closed = ski.morphology.closing(spec_array_binary, footprint_close)

                # footprint_open = ski.morphology.disk(2)
                # spec_array_opened = ski.morphology.opening(spec_array_closed, footprint_open)
                # spec_array_closed = spec_array_binary

                spec_array_labeled, num_blobs = label(spec_array_closed, structure=np.ones((3,3)))
                spec_array_processed = np.zeros_like(spec_array_labeled)

                # Extract the pitches by taking the left few pixels after the onset of the note and finding the centroid
                centroids = []
                plot_centroids = []
                for blob_index in range(num_blobs):
                    blob_array = spec_array_labeled == (blob_index+1)
                    blob_pixel_locs = np.argwhere(blob_array)  # Shape: (N_pixels, 2) (freq, time) I think

                    # Get the earliest onset
                    onset_time = np.min(blob_pixel_locs[:, 1])

                    # valid_pixels = blob_pixel_locs[blob_pixel_locs[:, 1] < onset_time + CENTROID_DETECTION_WIDTH]
                    valid_pixels = blob_pixel_locs
                    spec_array_processed[valid_pixels[:, 0], valid_pixels[:, 1]] = blob_index+1
                    
                    mean_freq = np.mean(valid_pixels[:, 0])
                    # print(f"Blob {blob_index} onset time {onset_time} with frequency {mean_freq}")

                    onset_time_s = processor.queue_index_to_sample(onset_time) / processor.rate
                    mean_freq_hz = processor.spec_y_to_freq(int(round(mean_freq)))
                    centroids.append((onset_time_s, mean_freq_hz))

                    plot_x = (processor.num_buffer_samples - num_current_frames) + onset_time
                    plot_y = mean_freq
                    plot_centroids.append((plot_x, plot_y))

                # print(centroids)
                
                # --- Update PROCESSED spectrogram plot (im2) ---
                full_spec_processed = np.full((N_MELS, processor.num_buffer_samples), 0)
                full_spec_processed[:, -num_current_frames:] = peaks_raw
                im2.set_data(full_spec_processed)
                # Adjust clim for the number of blobs
                vmax = np.max(full_spec_processed)
                # im2.set_clim(vmin=0, vmax=vmax if vmax > 0 else 1)
                im2.set_clim(vmin=np.min(peaks_raw), vmax=np.max(peaks_raw))

                if plot_centroids:
                    # Unzip the list of tuples into separate lists for x and y coordinates
                    x_coords, y_coords = zip(*plot_centroids)
                    centroid_plot_line.set_data(x_coords, y_coords)
                else:
                    # If no centroids are found, clear the plot
                    centroid_plot_line.set_data([], [])

            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(1.0 / REFRESH_RATE_HZ)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        traceback.print_exc()
    finally:
        print("Reached end")
        processor.close()
        plt.ioff()
        plt.show()
        print("Done.")


if __name__ == "__main__":
    main()