
# test_note_detection.py

import matplotlib.pyplot as plt
import numpy as np
import argparse
from audio_lib import AudioStreamProcessor
from scipy.ndimage import convolve, label
import skimage as ski
from pathlib import Path

data_path = Path(__file__).parent / "data"

# --- Constants for Plotting ---
BUFFER_DURATION = 5
REFRESH_RATE_HZ = 10

CENTROID_DETECTION_WIDTH = 4

def main():
    # --- Command-line Argument Parsing ---
    parser = argparse.ArgumentParser(description="Real-time audio pitch detection and visualization.")
    parser.add_argument(
        '--detector',
        type=str,
        choices=['yin', 'swiftf0'],
        default='yin',
        help="Pitch detection algorithm to use ('yin' or 'swiftf0')."
    )
    args = parser.parse_args()

    # Initialize the audio processor with the chosen detector
    print(f"Initializing with '{args.detector}' detector...")
    processor = AudioStreamProcessor(
        buffer_duration_s=BUFFER_DURATION,
        pitch_detector=args.detector,
        device_name="MacBook Pro Microphone",
        wav_filepath=data_path / "whistle_test.m4a"
    )

    # --- Matplotlib Setup (no changes here) ---
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))
    num_freq_bins = processor.CHUNK // 2 + 1
    initial_spectrogram = np.zeros((num_freq_bins, processor.NUM_BUFFER_CHUNKS))
    im = ax.imshow(initial_spectrogram, aspect='auto', origin='lower', cmap='viridis')
    fig.colorbar(im, ax=ax, format='%+2.0f dB')
    ax.set_title(f"Real-time Spectrogram and Pitch (Detector: {args.detector.upper()})") # <-- Updated title
    ax.set_xlabel("Time (chunks)")
    ax.set_ylabel("Frequency (Hz)")
    freqs = np.fft.rfftfreq(processor.CHUNK, 1./processor.RATE)
    tick_indices = np.linspace(0, len(freqs) - 1, num=10, dtype=int)
    ax.set_yticks(tick_indices)
    ax.set_yticklabels([f"{freqs[i]:.0f}" for i in tick_indices])
    line, = ax.plot([], [], 'r-', linewidth=2.5, label='Detected Pitch') # Made line thicker
    ax.legend()

    # Set the y limit to only show up to 5000 Hz for better visibility
    ax.set_ylim(0, 100)
    
    try:
        # The plotting loop remains IDENTICAL because our library abstracts the complexity.
        for spec_buffer, pitch_buffer in processor.listen():
            spectrogram_array = np.array(spec_buffer) # Shape: (time_chunks, freq_bins)
            # # Calculate the mean and std for the back half of the spectrogram buffer to set dynamic limits
            # spectrogram_array = np.array(spec_buffer)
            # if spectrogram_array.shape[1] >= processor.NUM_BUFFER_CHUNKS // 2:
            #     back_half = spectrogram_array[:, spectrogram_array.shape[1]//2 :]
            #     # Calculate the mean and std dev for each frequency bin
            #     mean_db = np.mean(back_half, axis=1)
            #     std_db = np.std(back_half, axis=1)
            #     # We only want to detect pitches that are more than 1 std dev above the mean db in each freq bin
            #     dynamic_thresholds = mean_db + std_db
            #     # Mask out data with dynamic thresholds (not used in plotting here, but could be)
            #     for i in range(spectrogram_array.shape[0]):
            #         spectrogram_array[i, spectrogram_array[i, :] < dynamic_thresholds[i]] = -80
            # else:
            #     dynamic_thresholds = np.ones(spectrogram_array.shape[0]) * 100  # High threshold if not enough data


            spectrogram_array = np.where(spectrogram_array < 0, -80, spectrogram_array)  # Clip very low dB values for better visualization
            if len(spec_buffer) > 0:
                spectrogram_data = spectrogram_array.T  # Shape: (freq_bins, time_chunks)

                # Apply a gaussian filter for smoothing in frequency
                gaussian_kernel = np.array([[1, 2, 1],
                                                [2, 4, 2],
                                                [1, 2, 1]]) / 16.0
                spectrogram_data = convolve(spectrogram_data, gaussian_kernel, mode='reflect')
                
                # # Apply x sobel filter for edge detection in time
                # sobel_kernel = np.array([[1, 0, -1],
                #                           [2, 0, -2],
                #                           [1, 0, -1]]) / 8.0
                # spectrogram_data = convolve(spectrogram_data, sobel_kernel, mode='reflect')

                spectrogram_data = spectrogram_data > 0

                footprint = ski.morphology.disk(6)
                spectrogram_data = ski.morphology.closing(spectrogram_data, footprint)

                # Label allows us to separated connected component efficiently
                spectrogram_data, num_blobs = label(spectrogram_data, structure=np.ones((3,3)))
                spectrogram_data_2 = np.zeros_like(spectrogram_data)

                # Extract the pitches by taking the left few pixels after the onset of the note and finding the centroid
                centroids = []
                for blob_index in range(num_blobs):
                    blob_array = spectrogram_data == (blob_index+1)
                    blob_pixel_locs = np.argwhere(blob_array)  # Shape: (N_pixels, 2) (freq, time) I think

                    # Get the earliest onset
                    onset_time = np.min(blob_pixel_locs[:, 1])

                    valid_pixels = blob_pixel_locs[blob_pixel_locs[:, 1] < onset_time + CENTROID_DETECTION_WIDTH]
                    spectrogram_data_2[valid_pixels[:, 0], valid_pixels[:, 1]] = blob_index+1
                    
                    mean_freq = np.mean(valid_pixels[:, 0])
                    # print(f"Blob {blob_index} onset time {onset_time} with frequency {mean_freq}")

                    centroids.append((onset_time, mean_freq))

                print(centroids)


                im.set_data(spectrogram_data_2)
                im.set_clim(vmin=np.min(spectrogram_data), vmax=np.max(spectrogram_data))
            # if len(pitch_buffer) > 0:
            #     pitches_hz = np.array(pitch_buffer)
            #     pitch_bins = np.array([np.argmin(np.abs(freqs - p)) if p > 0 else 0 for p in pitches_hz])
            #     time_coords = np.arange(len(pitch_bins))
            #     voiced_mask = pitch_bins > 0
            #     line.set_data(time_coords[voiced_mask], pitch_bins[voiced_mask])

            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(1.0 / REFRESH_RATE_HZ)

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        processor.close()
        plt.ioff()
        plt.show()
        print("Done.")

if __name__ == "__main__":
    main()


# # test_note_detection.py

# import matplotlib.pyplot as plt
# import numpy as np
# from audio_lib import AudioStreamProcessor

# # --- Constants for Plotting ---
# BUFFER_DURATION = 5  # seconds, should match the processor
# REFRESH_RATE_HZ = 10 # How many times per second to update the plot
# PITCH_DETECTOR = 'swiftf0'  # Using SwiftF0 for pitch detection

# def main():
#     # Initialize the audio processor
#     processor = AudioStreamProcessor(buffer_duration_s=BUFFER_DURATION, pitch_detector=PITCH_DETECTOR)
#     # --- Matplotlib Setup for Real-time Plotting ---
#     plt.ion() # Turn on interactive mode
#     fig, ax = plt.subplots(figsize=(12, 6))

#     # Create an initial empty spectrogram plot
#     # The number of frequency bins is CHUNK/2 + 1
#     num_freq_bins = processor.CHUNK // 2 + 1
#     initial_spectrogram = np.zeros((num_freq_bins, processor.NUM_BUFFER_CHUNKS))
    
#     # We use imshow for the spectrogram. 'origin=lower' places 0 Hz at the bottom.
#     # 'aspect=auto' allows the plot to fill the axes area.
#     im = ax.imshow(initial_spectrogram, aspect='auto', origin='lower', cmap='viridis')
#     fig.colorbar(im, ax=ax, format='%+2.0f dB')
#     ax.set_title("Real-time Spectrogram and Pitch Detection")
#     ax.set_xlabel("Time (chunks)")
#     ax.set_ylabel("Frequency (Hz)")

#     # Set the y-axis ticks to correspond to actual frequencies
#     freqs = np.fft.rfftfreq(processor.CHUNK, 1./processor.RATE)
#     # We can't show all frequency ticks, so we'll show a few spaced out ones.
#     tick_indices = np.linspace(0, len(freqs) - 1, num=10, dtype=int)
#     ax.set_yticks(tick_indices)
#     ax.set_yticklabels([f"{freqs[i]:.0f}" for i in tick_indices])

#     # Create an initial empty line plot for the pitch overlay
#     line, = ax.plot([], [], 'r-', linewidth=2, label='Detected Pitch (YIN)')
#     ax.legend()
    
#     try:
#         # Use the listen() generator to get data chunks
#         for spec_buffer, pitch_buffer in processor.listen():
            
#             # --- Update Spectrogram ---
#             if len(spec_buffer) > 0:
#                 # Transpose the buffer so time is on the x-axis
#                 spectrogram_data = np.array(spec_buffer).T
#                 im.set_data(spectrogram_data)
#                 # Rescale color limits based on current data for better visibility
#                 im.set_clim(vmin=np.min(spectrogram_data), vmax=np.max(spectrogram_data))

#             # --- Update Pitch Overlay ---
#             if len(pitch_buffer) > 0:
#                 pitches_hz = np.array(pitch_buffer)
                
#                 # Convert pitch in Hz to the y-axis pixel coordinates of the plot
#                 # We find the closest frequency bin for each detected pitch
#                 pitch_bins = np.array([np.argmin(np.abs(freqs - p)) if p > 0 else 0 for p in pitches_hz])
                
#                 # Create x-coordinates (time)
#                 time_coords = np.arange(len(pitch_bins))
                
#                 # Only plot voiced (non-zero) pitches
#                 voiced_mask = pitch_bins > 0
#                 line.set_data(time_coords[voiced_mask], pitch_bins[voiced_mask])

#                 # Update the title with the latest detected pitch
#                 if np.any(voiced_mask):
#                     latest_pitch = pitches_hz[voiced_mask][-1]
#                     ax.set_title(f"Real-time Spectrogram and Pitch Detection - Latest Pitch: {latest_pitch:.2f} Hz")
#                 else:
#                     ax.set_title("Real-time Spectrogram and Pitch Detection - Latest Pitch: None")

#             # Redraw the plot
#             fig.canvas.draw()
#             fig.canvas.flush_events()
#             plt.pause(1.0 / REFRESH_RATE_HZ)

#     except KeyboardInterrupt:
#         print("Interrupted by user.")
#     finally:
#         # Ensure the audio stream is closed properly
#         processor.close()
#         plt.ioff()
#         plt.show()
#         print("Done.")


# if __name__ == "__main__":
#     main()