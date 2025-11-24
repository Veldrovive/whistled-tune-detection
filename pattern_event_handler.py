import time
import traceback
from pathlib import Path
from typing import Callable, Dict, Optional

from audio_lib_v2 import AudioStreamProcessor
from pattern_recognition import PatternRecognizer

class PatternEventListener:
    def __init__(self, 
                 model_dir: str = "pattern_models",
                 buffer_duration: float = 5.0,
                 n_mels: int = 128,
                 refresh_rate_hz: int = 100,
                 fmin: int = 300,
                 fmax: int = 3500,
                 device_name: str = "MacBook Pro Microphone"):
        
        self.model_dir = Path(__file__).parent / model_dir
        self.refresh_rate_hz = refresh_rate_hz
        
        # Initialize Audio Processor
        self.processor = AudioStreamProcessor(
            chunk_size=1024,
            rate=22050,
            n_fft=2048,
            hop_len=512,
            buffer_duration_s=buffer_duration,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            max_curve_jump=2,
            min_interesting_curve_len=5,
            wav_filepath=None,
            device_name=device_name
        )
        
        # Initialize Pattern Recognizer
        self.recognizer = PatternRecognizer(self.model_dir)
        print(f"PatternEventListener initialized. Loaded models: {list(self.recognizer.models.keys())}")
        
        # Callback storage
        self.callbacks: Dict[str, Callable] = {}
        self.running = False

    def on(self, pattern_name: str, callback: Callable):
        """Register a callback function for a specific pattern."""
        self.callbacks[pattern_name] = callback
        print(f"Registered callback for pattern: '{pattern_name}'")

    def start(self):
        """Start listening for patterns and triggering callbacks. This is a blocking call."""
        self.running = True
        print("Starting PatternEventListener loop...")
        
        try:
            for spec_buffer, new_curves in self.processor.listen():
                if not self.running:
                    break
                
                # Check for patterns in new curves
                for curve in new_curves:
                    # print(f"New curve: {curve}. Total curves: {len(self.processor.finished_curves)}")
                    detections = self.recognizer.recognize(curve, self.processor.finished_curves)
                    
                    if detections:
                        print(f"Detected patterns: {detections}")
                        for detection in detections:
                            pattern_name = detection['name']
                            score = detection['score']
                            
                            if pattern_name in self.callbacks:
                                try:
                                    # Call the user's function
                                    self.callbacks[pattern_name](detection)
                                except Exception as e:
                                    print(f"Error in callback for '{pattern_name}': {e}")
                                    traceback.print_exc()
                            else:
                                print(f"No callback registered for pattern: '{pattern_name}'")
                
                # Sleep to maintain refresh rate (approximate)
                # time.sleep(1 / self.refresh_rate_hz)
                
        except KeyboardInterrupt:
            print("\nPatternEventListener stopped by user.")
        except Exception as e:
            print(f"Error in PatternEventListener loop: {e}")
            traceback.print_exc()
        finally:
            self.stop()

    def stop(self):
        """Stop the listener and cleanup resources."""
        self.running = False
        self.processor.close()
        print("PatternEventListener closed.")
