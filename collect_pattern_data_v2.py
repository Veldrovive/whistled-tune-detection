import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
from audio_lib_v2 import AudioStreamProcessor, Curve
import traceback
import argparse
import pickle
import os
import time
from pathlib import Path

BUFFER_DURATION = 5
N_MELS = 128
REFRESH_RATE_HZ = 10
FMIN = 300
FMAX = 3500
PATTERN_SILENCE_SEPARATION = 1 # Seconds of silence to trigger new pattern
MIN_PATTERN_LENGTH = 3 # Minimum number of notes in a pattern
PATTERN_DATA_DIR = Path(__file__).parent / "pattern_data"
PATTERN_DATA_DIR.mkdir(exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="Collect whistled pattern data.")
    parser.add_argument("pattern_name", type=str, help="Name of the pattern to save.")
    args = parser.parse_args()
    
    pattern_name = args.pattern_name
    filename = PATTERN_DATA_DIR / f"{pattern_name}_pattern_data.pkl"
    
    print(f"Collecting data for pattern: {pattern_name}")
    print(f"Saving to: {filename}")
    
    processor = AudioStreamProcessor(
        chunk_size=1024,
        rate=22050,
        n_fft=2048,
        hop_len=512,
        buffer_duration_s=BUFFER_DURATION,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        max_curve_jump=2,
        min_interesting_curve_len=5,
        wav_filepath=None,
        device_name="MacBook Pro Microphone"
    )

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Initial Spectrogram
    initial_spectrogram = np.zeros((N_MELS, processor.num_buffer_samples))
    im1 = ax1.imshow(initial_spectrogram, aspect='auto', origin='lower', cmap='viridis', vmin=-80, vmax=0)
    fig.colorbar(im1, ax=ax1, format='%+2.0f dB')
    ax1.set_title(f"Recording Pattern: {pattern_name}")
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
    
    # Visualization lines
    pattern_lines = [] # List of vertical line objects

    try:
        print(f"Listening... Whistle your pattern. Pause for >{PATTERN_SILENCE_SEPARATION}s to finish a pattern.")
        last_curve_end_time = None
        current_pattern = []
        verified_patterns = []
        unverified_pattern = None
        
        # Verify button
        verify_button = Button(ax1, "Verify Pattern")

        def verify_pattern(event):
            nonlocal unverified_pattern
            if unverified_pattern is not None:
                verified_patterns.append(unverified_pattern)
                unverified_pattern = None
                
        verify_button.on_clicked(verify_pattern)
        
        def get_x_from_time(t):
            if len(processor.spectrogram_times) > 0:
                # Convert deque to list for searching
                times = list(processor.spectrogram_times)
                # Find closest
                idx = (np.abs(np.array(times) - t)).argmin()
                
                # Check if the time difference is reasonable (e.g. within buffer duration)
                if abs(times[idx] - t) < 1.0:
                     L = len(times)
                     return processor.num_buffer_samples - L + idx
            return None

        for spec_buffer, new_curves in processor.listen():
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

                # 3. Process new Curves
                current_time = processor.total_samples_processed / processor.rate
                # In wall-clock mode, total_samples_processed might not reflect wall time exactly if we used that for current_time logic?
                # Actually, in audio_lib_v2, we updated total_samples_processed by chunk_size.
                # But we also have spectrogram_times[-1] which is the most accurate "current time" of the latest frame.
                if len(processor.spectrogram_times) > 0:
                    current_time = processor.spectrogram_times[-1]
                
                if last_curve_end_time is not None and current_time - last_curve_end_time > PATTERN_SILENCE_SEPARATION:
                    # Then we have finished a pattern
                    if len(current_pattern) >= MIN_PATTERN_LENGTH:
                        unverified_pattern = current_pattern.copy()
                    current_pattern = []
                    last_curve_end_time = None

                for curve in new_curves:
                    current_pattern.append(curve)
                    last_curve_end_time = curve.points[-1][0]

                # 4. Visualize Notes and Patterns

                # Collect notes to plot
                note_plot_x = []
                note_plot_y = []

                # Only plot notes that will be visible
                visible_curves = []
                for curve in processor.finished_curves:
                    if curve.points[0][0] > (current_time - BUFFER_DURATION):
                        visible_curves.append(curve)
                
                for curve in visible_curves:
                    onset_time = curve.points[0][0]
                    freqs_in_curve = [p[1] for p in curve.points]
                    avg_freq = np.mean(freqs_in_curve)
                    y_pos = np.argmin(np.abs(processor.mel_center_freqs - avg_freq))
                    
                    x_pos = get_x_from_time(onset_time)
                    if x_pos is not None:
                        note_plot_x.append(x_pos)
                        note_plot_y.append(y_pos)

                note_scatter.set_offsets(np.column_stack((note_plot_x, note_plot_y)))

                # Plot vertical lines for patterns
                for line in pattern_lines:
                    line.remove()
                pattern_lines.clear()
                
                visible_pattern_info = []
                if unverified_pattern is not None:
                    p_start = unverified_pattern[0].points[0][0]
                    p_end = unverified_pattern[-1].points[-1][0]
                    visible_pattern_info.append((p_start, p_end, "#FF0000"))
                
                for pattern in verified_patterns:
                    if pattern[0].points[0][0] > (current_time - BUFFER_DURATION):
                        visible_pattern_info.append((pattern[0].points[0][0], pattern[-1].points[-1][0], "#000000"))

                for start, end, color in visible_pattern_info:
                    if start > (current_time - BUFFER_DURATION):
                        x_pos = get_x_from_time(start)
                        if x_pos is not None:
                            line = ax1.axvline(x=x_pos, color=color, linestyle="--", alpha=0.7)
                            pattern_lines.append(line)

                    if end > (current_time - BUFFER_DURATION):
                        x_pos = get_x_from_time(end)
                        if x_pos is not None:
                            line = ax1.axvline(x=x_pos, color=color, linestyle="-", alpha=0.7)
                            pattern_lines.append(line)
                        
            fig.canvas.draw()
            fig.canvas.flush_events()
            # plt.pause(0.01)

    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        traceback.print_exc()
    finally:
        print("Reached end")
        processor.close()
        plt.ioff()
        plt.close()
        
        # Save Data
        if current_pattern:
            # verified_patterns.append(current_pattern) # Don't auto-save unverified
            pass
            
        # Filter empty patterns
        verified_patterns = [p for p in verified_patterns if p]

        # Filter patterns that are too short
        verified_patterns = [p for p in verified_patterns if len(p) >= MIN_PATTERN_LENGTH]
        
        print(f"Saving {len(verified_patterns)} patterns to {filename}...")
        with open(filename, 'wb') as f:
            pickle.dump(verified_patterns, f)
        print("Done.")

if __name__ == "__main__":
    main()
