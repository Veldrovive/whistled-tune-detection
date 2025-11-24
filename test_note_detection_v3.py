import matplotlib.pyplot as plt
import numpy as np
from audio_lib_v2 import AudioStreamProcessor
import traceback

BUFFER_DURATION = 5
N_MELS = 128
REFRESH_RATE_HZ = 10
FMIN = 300
FMAX = 3500

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
        wav_filepath=None,
        device_name="MacBook Pro Microphone"
    )

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Initial Spectrogram
    initial_spectrogram = np.zeros((N_MELS, processor.num_buffer_samples))
    im1 = ax1.imshow(initial_spectrogram, aspect='auto', origin='lower', cmap='viridis', vmin=-80, vmax=0)
    fig.colorbar(im1, ax=ax1, format='%+2.0f dB')
    ax1.set_title("Real-time Spectrogram & Detected Notes")
    ax1.set_ylabel("Frequency (Hz)")
    
    # Overlay for detected notes
    note_scatter = ax1.scatter([], [], c='red', s=50, marker='x', label='Detected Notes')
    ax1.legend(loc='upper right')

    # Peaks Plot
    im2 = ax2.imshow(initial_spectrogram, aspect='auto', origin='lower', cmap='inferno', vmin=0, vmax=50)
    fig.colorbar(im2, ax=ax2)
    ax2.set_title("Detected Peaks (Prominence)")
    ax2.set_ylabel("Frequency (Hz)")
    ax2.set_xlabel("Time (frames)")

    # Frequency ticks
    freqs = processor.mel_center_freqs
    tick_indices = np.linspace(0, len(freqs) - 1, num=10, dtype=int)
    for ax in [ax1, ax2]:
        ax.set_yticks(tick_indices)
        ax.set_yticklabels([f"{freqs[i]:.0f}" for i in tick_indices])

    try:
        print("Listening...")
        for spec_buffer, _ in processor.listen():
            if len(spec_buffer) > 0:
                # 1. Update Spectrogram
                spec_array_db = np.array(spec_buffer).T
                num_current_frames = spec_array_db.shape[1]
                
                full_spec = np.full((N_MELS, processor.num_buffer_samples), -80.0)
                full_spec[:, -num_current_frames:] = spec_array_db
                im1.set_data(full_spec)
                im1.set_clim(vmin=-60, vmax=np.max(full_spec) if np.max(full_spec) > -60 else 0)

                # 2. Update Peaks Plot
                peaks_raw = np.zeros((N_MELS, len(processor.peaks_buffer)), dtype=np.float32)
                for i, (peaks, properties) in enumerate(processor.peaks_buffer):
                    if len(peaks) > 0:
                        peaks_raw[peaks, i] = properties["prominences"]
                
                full_peaks = np.zeros((N_MELS, processor.num_buffer_samples), dtype=np.float32)
                full_peaks[:, -len(processor.peaks_buffer):] = peaks_raw
                im2.set_data(full_peaks)
                im2.set_clim(vmin=0, vmax=np.max(full_peaks) if np.max(full_peaks) > 0 else 1)

                # 3. Process Curves for Visualization
                current_time = processor.total_samples_processed / processor.rate
                
                # Collect points to plot
                plot_x = []
                plot_y = []
                
                # We want to plot finished curves that are still within the buffer window
                # And maybe active curves too? User asked for "detected notes", usually implies finished.
                # Let's plot finished curves.
                
                visible_curves = []
                for curve in processor.finished_curves:
                    # Calculate Onset and Centroid
                    onset_time = curve.points[0][0]
                    
                    # Check if it's in the buffer window
                    # Window start: current_time - BUFFER_DURATION
                    if onset_time > (current_time - BUFFER_DURATION):
                        visible_curves.append(curve)
                
                for curve in visible_curves:
                    onset_time = curve.points[0][0]
                    
                    # Centroid Frequency (Average of frequencies in the curve)
                    freqs_in_curve = [p[1] for p in curve.points]
                    avg_freq = np.mean(freqs_in_curve)
                    
                    # Map freq to y-index (mel bin)
                    # We need to find the closest mel bin index for this frequency
                    # processor.mel_center_freqs is the array of frequencies
                    y_pos = np.argmin(np.abs(processor.mel_center_freqs - avg_freq))
                    
                    # Map time to x-index
                    # time_diff = current_time - onset_time
                    # frames_ago = time_diff * rate / hop_len
                    # x_pos = num_buffer_samples - frames_ago
                    
                    time_diff = current_time - onset_time
                    frames_ago = time_diff * processor.rate / processor.hop_len
                    x_pos = processor.num_buffer_samples - frames_ago
                    
                    plot_x.append(x_pos)
                    plot_y.append(y_pos)
                
                note_scatter.set_offsets(np.c_[plot_x, plot_y])

            fig.canvas.draw()
            fig.canvas.flush_events()
            # plt.pause(1.0 / REFRESH_RATE_HZ) 
            # Using a small pause to allow GUI update
            plt.pause(0.01)

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
