import matplotlib.pyplot as plt
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
PATTERN_SILENCE_SEPARATION = 0.5 # Seconds of silence to trigger new pattern
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

    # Pattern Collection State
    all_patterns = [] # List of lists of Curves
    current_pattern = [] # List of Curves
    last_curve_end_time = -1.0
    
    # Visualization lines
    pattern_lines = [] # List of vertical line objects

    try:
        print("Listening... Whistle your pattern. Pause for >0.5s to finish a pattern.")
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

                # 3. Process Curves & Patterns
                current_time = processor.total_samples_processed / processor.rate
                
                # Check for finished curves that we haven't processed yet
                # We can't easily know which ones are "new" in finished_curves without tracking index
                # So let's keep a local set of processed curve IDs or just check the last few
                # Actually, finished_curves grows. We can just iterate from where we left off.
                # But processor.finished_curves is a list.
                # Let's just iterate all and filter by time to find new ones? 
                # Or better, clear finished_curves in processor? No, might break other things.
                # Let's keep a local index.
                
                # Hack: we will just look at curves that finished "recently" and see if we added them.
                # But since we are building `current_pattern`, we can check if curve is in `current_pattern` or `all_patterns`.
                # `Curve` objects are unique instances.
                
                # Optimization: Only check the last N curves
                recent_curves = processor.finished_curves[-20:] if len(processor.finished_curves) > 20 else processor.finished_curves
                
                for curve in recent_curves:
                    # Check if already processed
                    is_processed = False
                    if curve in current_pattern:
                        is_processed = True
                    else:
                        for p in all_patterns[-5:]: # Check last few patterns
                            if curve in p:
                                is_processed = True
                                break
                    
                    if is_processed:
                        continue
                        
                    # It's a new curve!
                    curve_start = curve.points[0][0]
                    curve_end = curve.points[-1][0]
                    
                    # Check for silence gap
                    if last_curve_end_time > 0 and (curve_start - last_curve_end_time) > PATTERN_SILENCE_SEPARATION:
                        # Pattern finished!
                        if current_pattern:
                            print(f"Pattern detected! ({len(current_pattern)} notes)")
                            all_patterns.append(current_pattern)
                            current_pattern = []
                            
                    current_pattern.append(curve)
                    last_curve_end_time = curve_end
                
                # Check if current pattern is finished due to silence (even if no new note comes)
                # If we are waiting for a note and time passes...
                # Actually, we only "finalize" a pattern when the NEXT note starts or when we quit.
                # But for visualization, we might want to show "Pattern Finished" marker if enough time passed.
                # Let's stick to the logic: Pattern ends when NEXT note starts after gap.
                # OR we can finalize if silence > threshold.
                if current_pattern and (current_time - last_curve_end_time) > PATTERN_SILENCE_SEPARATION:
                     # This logic would split it immediately.
                     # Let's do it.
                     print(f"Pattern detected (timeout)! ({len(current_pattern)} notes)")
                     all_patterns.append(current_pattern)
                     current_pattern = []

                # 4. Visualization of Notes and Patterns
                
                # Collect points to plot (Notes)
                plot_x = []
                plot_y = []
                
                # Plot finished curves in buffer
                visible_curves = []
                for curve in processor.finished_curves:
                    if curve.points[0][0] > (current_time - BUFFER_DURATION):
                        visible_curves.append(curve)
                
                for curve in visible_curves:
                    onset_time = curve.points[0][0]
                    freqs_in_curve = [p[1] for p in curve.points]
                    avg_freq = np.mean(freqs_in_curve)
                    y_pos = np.argmin(np.abs(processor.mel_center_freqs - avg_freq))
                    time_diff = current_time - onset_time
                    frames_ago = time_diff * processor.rate / processor.hop_len
                    x_pos = processor.num_buffer_samples - frames_ago
                    plot_x.append(x_pos)
                    plot_y.append(y_pos)
                
                note_scatter.set_offsets(np.c_[plot_x, plot_y])
                
                # Plot Vertical Lines for Patterns
                # Clear old lines
                for line in pattern_lines:
                    line.remove()
                pattern_lines = []
                
                # Find patterns visible in window
                visible_patterns = []
                # Check all patterns? Might be slow if many.
                # Check last 10
                for pattern in all_patterns[-10:] + ([current_pattern] if current_pattern else []):
                    if not pattern: continue
                    p_start = pattern[0].points[0][0]
                    p_end = pattern[-1].points[-1][0]
                    
                    if p_end > (current_time - BUFFER_DURATION):
                        visible_patterns.append((p_start, p_end))
                
                for p_start, p_end in visible_patterns:
                    # Start Line
                    if p_start > (current_time - BUFFER_DURATION):
                        time_diff = current_time - p_start
                        frames_ago = time_diff * processor.rate / processor.hop_len
                        x_pos = processor.num_buffer_samples - frames_ago
                        line = ax1.axvline(x=x_pos, color='white', linestyle='--', alpha=0.7)
                        pattern_lines.append(line)
                        
                    # End Line
                    if p_end > (current_time - BUFFER_DURATION):
                        time_diff = current_time - p_end
                        frames_ago = time_diff * processor.rate / processor.hop_len
                        x_pos = processor.num_buffer_samples - frames_ago
                        line = ax1.axvline(x=x_pos, color='white', linestyle='-', alpha=0.7)
                        pattern_lines.append(line)

            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.01)

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
            all_patterns.append(current_pattern)
            
        # Filter empty patterns
        all_patterns = [p for p in all_patterns if p]

        # Filter patterns that are too short
        all_patterns = [p for p in all_patterns if len(p) >= MIN_PATTERN_LENGTH]
        
        print(f"Saving {len(all_patterns)} patterns to {filename}...")
        with open(filename, 'wb') as f:
            pickle.dump(all_patterns, f)
        print("Done.")

if __name__ == "__main__":
    main()
