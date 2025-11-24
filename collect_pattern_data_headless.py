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
FMIN = 300
FMAX = 3500
PATTERN_SILENCE_SEPARATION = 1.0 # Seconds of silence to trigger new pattern
MIN_PATTERN_LENGTH = 3 # Minimum number of notes in a pattern
MAX_CURVE_JUMP = 2 # Maximum frequency jump between notes in a pattern
MIN_INTERESTING_CURVE_LEN = 10 # Minimum number of points in a curve to consider it interesting
PATTERN_DATA_DIR = Path(__file__).parent / "pattern_data"
PATTERN_DATA_DIR.mkdir(exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="Collect whistled pattern data (Headless).")
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
        max_curve_jump=MAX_CURVE_JUMP,
        min_interesting_curve_len=MIN_INTERESTING_CURVE_LEN,
        wav_filepath=None,
        device_name="MacBook Pro Microphone"
    )

    try:
        print(f"Listening... Whistle your pattern. Pause for >{PATTERN_SILENCE_SEPARATION}s to finish a pattern.")
        print("Press Ctrl+C to stop and save.")
        
        last_curve_end_time = None
        current_pattern = []
        verified_patterns = []
        
        for spec_buffer, new_curves in processor.listen():
            # Process new Curves
            current_time = processor.total_samples_processed / processor.rate
            
            if last_curve_end_time is not None and current_time - last_curve_end_time > PATTERN_SILENCE_SEPARATION:
                # Then we have finished a pattern
                if len(current_pattern) >= MIN_PATTERN_LENGTH:
                    print(f"\nCaptured pattern with {len(current_pattern)} curves.")
                    
                    # Pause and ask for verification
                    while True:
                        choice = input("Keep this pattern? (y/n/q): ").lower().strip()
                        if choice == 'y':
                            verified_patterns.append(current_pattern)
                            print("Pattern saved.")
                            break
                        elif choice == 'n':
                            print("Pattern discarded.")
                            break
                        elif choice == 'q':
                            print("Quitting...")
                            raise KeyboardInterrupt # Trigger save and exit
                        
                    print("Resuming listening...\n")
                    
                elif len(current_pattern) > 0:
                    print(f"Discarded short pattern ({len(current_pattern)} curves)")
                    
                current_pattern = []
                last_curve_end_time = None

            for curve in new_curves:
                current_pattern.append(curve)
                last_curve_end_time = curve.points[-1][0]
                print(f"Curve: {curve}")

    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        traceback.print_exc()
    finally:
        print("Reached end")
        processor.close()
        
        # Save Data
        if current_pattern and len(current_pattern) >= MIN_PATTERN_LENGTH:
            verified_patterns.append(current_pattern)
            
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
