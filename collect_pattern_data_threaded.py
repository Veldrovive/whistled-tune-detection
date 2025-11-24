import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
from audio_lib_v3 import AudioStreamProcessorThreaded, Curve
import traceback
import argparse
import pickle
import os
import time
from pathlib import Path

BUFFER_DURATION = 5
N_MELS = 128
FMIN = 300
FMAX = 3500
PATTERN_SILENCE_SEPARATION = 1.0 # Seconds of silence to trigger new pattern
MIN_PATTERN_LENGTH = 3 # Minimum number of notes in a pattern
MAX_CURVE_JUMP = 2
MIN_INTERESTING_CURVE_LEN = 10
PATTERN_DATA_DIR = Path(__file__).parent / "pattern_data"
PATTERN_DATA_DIR.mkdir(exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="Collect whistled pattern data (Threaded).")
    parser.add_argument("pattern_name", type=str, help="Name of the pattern to save.")
    args = parser.parse_args()
    
    pattern_name = args.pattern_name
    filename = PATTERN_DATA_DIR / f"{pattern_name}_pattern_data.pkl"
    
    print(f"Collecting data for pattern: {pattern_name}")
    print(f"Saving to: {filename}")
    
    processor = AudioStreamProcessorThreaded(
        chunk_size=1024,
        rate=22050,
        n_fft=2048,
        hop_len=512,
        buffer_duration_s=BUFFER_DURATION,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        max_curve_jump=MAX_CURVE_JUMP,
        min_interesting_curve_len=MIN_INTERESTING_CURVE_LEN,
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
        print("Press Ctrl+C to stop and save.")
        
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
                print(f"Pattern verified! Total saved: {len(verified_patterns)}")
                unverified_pattern = None
                
        verify_button.on_clicked(verify_pattern)
        
        # Main loop consuming data from the audio thread
        for recent_spec, recent_peaks, new_curves, total_samples in processor.listen():
            
            # 1. Update Spectrogram Data
            # We need to append the new data to our local full buffer
            # Shift buffer
            num_new_frames = len(recent_spec)
            if num_new_frames > 0:
                # Roll the spectrogram
                initial_spectrogram = np.roll(initial_spectrogram, -num_new_frames, axis=1)
                # Update the end with new data
                new_spec_data = np.array(recent_spec).T
                initial_spectrogram[:, -num_new_frames:] = new_spec_data
                im1.set_data(initial_spectrogram)
                im1.set_clim(vmin=-60, vmax=np.max(initial_spectrogram) if np.max(initial_spectrogram) > -60 else 0)

                # Update Peaks
                # This is a bit trickier since peaks are sparse.
                # We'll just rebuild the peaks image for the new frames and roll it.
                # Actually, im2.get_array() returns the data.
                current_peaks_img = im2.get_array()
                current_peaks_img = np.roll(current_peaks_img, -num_new_frames, axis=1)
                current_peaks_img[:, -num_new_frames:] = 0 # Clear new area
                
                for i, (peaks, properties) in enumerate(recent_peaks):
                    if len(peaks) > 0:
                        # The index i is relative to the start of recent_peaks
                        # We want to place it at the end of the buffer
                        col_idx = -num_new_frames + i
                        current_peaks_img[peaks, col_idx] = properties["prominences"]
                
                im2.set_data(current_peaks_img)
                im2.set_clim(vmin=0, vmax=np.max(current_peaks_img) if np.max(current_peaks_img) > 0 else 1)

            # 2. Process new Curves
            current_time = total_samples / processor.rate
            
            if last_curve_end_time is not None and current_time - last_curve_end_time > PATTERN_SILENCE_SEPARATION:
                # Then we have finished a pattern
                if len(current_pattern) >= MIN_PATTERN_LENGTH:
                    print(f"Captured pattern with {len(current_pattern)} curves. Click 'Verify Pattern' to keep it.")
                    unverified_pattern = current_pattern.copy()
                elif len(current_pattern) > 0:
                    print(f"Discarded short pattern ({len(current_pattern)} curves)")
                    
                current_pattern = []
                last_curve_end_time = None

            for curve in new_curves:
                current_pattern.append(curve)
                last_curve_end_time = curve.points[-1][0]
                print(f"Curve: {curve}")

            # 3. Visualize Notes and Patterns
            # Only plot notes that will be visible
            note_plot_x = []
            note_plot_y = []
            
            visible_curves = []
            # We need access to all finished curves. processor.finished_curves is updated in the thread.
            # But we only get new_curves here. 
            # We should maintain a local list of finished curves for visualization.
            # Actually, processor.finished_curves IS available because we share the object instance,
            # BUT accessing it while the thread writes to it is not strictly thread-safe.
            # However, for visualization, it's usually "okay" if we miss a frame.
            # A safer way is to accumulate new_curves into a local list.
            
            # Let's use a local list for visualization
            if not hasattr(main, "local_finished_curves"):
                main.local_finished_curves = []
            main.local_finished_curves.extend(new_curves)
            
            for curve in main.local_finished_curves:
                if curve.points[0][0] > (current_time - BUFFER_DURATION):
                    visible_curves.append(curve)
            
            # Prune old curves from local list to prevent memory growth
            main.local_finished_curves = visible_curves 
            
            for curve in visible_curves:
                onset_time = curve.points[0][0]
                freqs_in_curve = [p[1] for p in curve.points]
                avg_freq = np.mean(freqs_in_curve)
                y_pos = np.argmin(np.abs(processor.mel_center_freqs - avg_freq))
                time_diff = current_time - onset_time
                frames_ago = time_diff * processor.rate / processor.hop_len
                x_pos = processor.num_buffer_samples - frames_ago
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
                    time_diff = current_time - start
                    frames_ago = time_diff * processor.rate / processor.hop_len
                    x_pos = processor.num_buffer_samples - frames_ago
                    line = ax1.axvline(x=x_pos, color=color, linestyle="--", alpha=0.7)
                    pattern_lines.append(line)

                if end > (current_time - BUFFER_DURATION):
                    time_diff = current_time - end
                    frames_ago = time_diff * processor.rate / processor.hop_len
                    x_pos = processor.num_buffer_samples - frames_ago
                    line = ax1.axvline(x=x_pos, color=color, linestyle="-", alpha=0.7)
                    pattern_lines.append(line)
                        
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            # No sleep needed here! The queue.get() acts as the pacer.

    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        traceback.print_exc()
    finally:
        print("Reached end")
        processor.stop()
        plt.ioff()
        plt.close()
        
        # Save Data
        if unverified_pattern:
             # Ask one last time? Nah, just save verified ones.
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
