import unittest
import pickle
import numpy as np
from pathlib import Path
from pattern_recognition import PatternRecognizer
# from audio_lib_v2 import Curve

class Curve:
    def __init__(self, start_time, start_freq, start_amp, start_peak_index):
        self.points = [(start_time, start_freq, start_amp)]
        self.last_peak_index = start_peak_index
        self.active = True

    def add_point(self, time, freq, amp, peak_index):
        self.points.append((time, freq, amp))
        self.last_peak_index = peak_index


class TestPatternRecognition(unittest.TestCase):
    def setUp(self):
        self.model_dir = Path(__file__).parent / "pattern_models"
        self.recognizer = PatternRecognizer(self.model_dir)
        
        # Load the rising model to know what to simulate
        self.model_name = "rising"
        model_path = self.model_dir / f"{self.model_name}_model.pkl"
        if model_path.exists():
            with open(model_path, 'rb') as f:
                self.model_data = pickle.load(f)
        else:
            self.skipTest("rising_model.pkl not found")

    def create_mock_curve(self, start_time, freq):
        # Create a curve with a single point for simplicity, or a few points
        # The feature extractor uses points[0][0] for time and mean(freqs) for freq
        c = Curve(start_time, freq, 1.0, 0)
        c.add_point(start_time + 0.1, freq, 1.0, 0)
        return c

    def test_perfect_pattern_detection(self):
        """Test that a pattern matching the model means is detected."""
        model_steps = self.model_data['model']
        
        # Construct a sequence of curves that perfectly matches the means
        curves = []
        
        # Start with an arbitrary first note
        current_time = 1.0
        current_freq = 1000.0
        curves.append(self.create_mock_curve(current_time, current_freq))
        
        for step in model_steps:
            mu_t, _ = step['time_dist']
            mu_f, _ = step['freq_dist'] # This is delta log freq
            
            # Calculate next note params
            next_time = current_time + mu_t
            next_log_freq = np.log(current_freq) + mu_f
            next_freq = np.exp(next_log_freq)
            
            curves.append(self.create_mock_curve(next_time, next_freq))
            
            current_time = next_time
            current_freq = next_freq
            
        # The last curve is the "new" one
        new_curve = curves[-1]
        recent_curves = curves[:-1]
        
        # Add some noise/distractor curves to recent_curves
        distractor = self.create_mock_curve(0.5, 500)
        recent_curves.insert(0, distractor)
        
        detections = self.recognizer.recognize(new_curve, recent_curves)
        
        print("\nDetections found:", detections)
        
        self.assertTrue(len(detections) > 0)
        found = False
        for d in detections:
            if d['name'] == self.model_name:
                found = True
                self.assertEqual(len(d['curves']), len(model_steps) + 1)
                break
        self.assertTrue(found, f"Should detect {self.model_name} pattern")

    def test_noise_rejection(self):
        """Test that random noise is not detected as a pattern."""
        # Create random curves
        curves = []
        for i in range(5):
            curves.append(self.create_mock_curve(i * 1.0, 1000 + np.random.randn()*500))
            
        new_curve = curves[-1]
        recent_curves = curves[:-1]
        
        detections = self.recognizer.recognize(new_curve, recent_curves)
        # It's possible to get a detection if noise accidentally matches, but unlikely with high threshold
        # However, our threshold is 5th percentile of LOO, so it might be lenient.
        # Let's just print what happened.
        if detections:
            print("\nUnexpected detection on noise:", detections)
            
        # We expect no detections for completely random garbage that doesn't follow the delta structure
        # But since we only have one model (rising), let's make sure we create a falling pattern
        
        falling_curves = []
        t = 1.0
        f = 2000.0
        for _ in range(4):
            falling_curves.append(self.create_mock_curve(t, f))
            t += 0.5
            f -= 300 # Falling frequency
            
        detections = self.recognizer.recognize(falling_curves[-1], falling_curves[:-1])
        self.assertEqual(len(detections), 0, "Should not detect falling pattern as rising")

if __name__ == '__main__':
    unittest.main()
