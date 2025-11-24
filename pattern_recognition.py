import pickle
import numpy as np
import math
from pathlib import Path
from scipy.stats import norm

class PatternRecognizer:
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.models = self._load_models()

    def _load_models(self):
        models = {}
        if not self.model_dir.exists():
            print(f"Warning: Model directory {self.model_dir} does not exist.")
            return models

        for model_file in self.model_dir.glob("*_model.pkl"):
            try:
                with open(model_file, 'rb') as f:
                    data = pickle.load(f)
                    # data structure expected: {'model': [...], 'loo_scores': [...], 'pattern_length': int}
                    pattern_name = model_file.stem.replace("_model", "")
                    models[pattern_name] = data
                    print(f"Loaded model: {pattern_name}")
            except Exception as e:
                print(f"Failed to load model {model_file}: {e}")
        return models

    def _extract_features(self, curve):
        """
        Extracts (onset_time, centroid_freq) from a Curve object.
        """
        # Assuming curve has .points list of (time, freq, amp)
        if not hasattr(curve, 'points') or not curve.points:
            return None
        
        onset_time = curve.points[0][0]
        freqs = [p[1] for p in curve.points]
        centroid_freq = np.mean(freqs)
        return (onset_time, centroid_freq)

    def _calculate_step_log_prob(self, prev_feat, curr_feat, step_model):
        """
        Calculates log probability of the transition from prev_feat to curr_feat
        given the step_model (Gaussian params for time and freq deltas).
        """
        t1, f1 = prev_feat
        t2, f2 = curr_feat
        
        delta_time = t2 - t1
        # Avoid log(0) or negative freq if that happens, though unlikely for valid whistles
        if f1 <= 0 or f2 <= 0:
            return -np.inf
            
        delta_log_freq = math.log(f2) - math.log(f1)
        
        mu_t, sigma_t = step_model['time_dist']
        mu_f, sigma_f = step_model['freq_dist']

        # print(f"Delta time: {delta_time:.2f} ({t1:.2f} -> {t2:.2f}) with mu {mu_t:.2f} and sigma {sigma_t:.2f}, Delta log freq: {delta_log_freq:.2f} ({f1:.2f} -> {f2:.2f}) with mu {mu_f:.2f} and sigma {sigma_f:.2f}")
        
        # Add epsilon to sigma to avoid division by zero
        sigma_t = max(sigma_t, 1e-6)
        sigma_f = max(sigma_f, 1e-6)
        
        lp_t = norm.logpdf(delta_time, loc=mu_t, scale=sigma_t)
        lp_f = norm.logpdf(delta_log_freq, loc=mu_f, scale=sigma_f)
        
        return lp_t + lp_f

    def recognize(self, new_curve, recent_curves, search_window=20):
        """
        Attempts to recognize patterns ending with new_curve.
        
        Args:
            new_curve: The most recently finished curve.
            recent_curves: List of previously finished curves (excluding new_curve).
                           Should be sorted by time (oldest to newest) ideally, 
                           but we will iterate backwards.
            search_window: Max number of recent curves to check for each step.
            
        Returns:
            List of detected patterns: [(pattern_name, score, [curve_objects])]
        """
        detections = []
        
        new_curve_feat = self._extract_features(new_curve)
        if new_curve_feat is None:
            return detections

        # Pre-extract features for recent curves to avoid re-computation
        # We iterate backwards, so let's reverse the list for easier access
        # recent_curves_rev = list(reversed(recent_curves))
        # Actually, let's just keep them as is and iterate with index.
        
        recent_feats = []
        valid_recent_curves = []
        for c in recent_curves:
            f = self._extract_features(c)
            if f:
                recent_feats.append(f)
                valid_recent_curves.append(c)
        
        # We need to search backwards.
        # Let's convert to a list we can slice easily.
        
        for name, data in self.models.items():
            model_steps = data['model'] # List of step models
            pattern_length = data['pattern_length']
            loo_scores = data['loo_scores']
            
            # Threshold: e.g., mean - 2*std of LOO scores, or just a raw probability check.
            # The user said: "If it is comparably large compared to those recorded in the pattern model"
            # Let's use the 5th percentile of LOO scores as a loose threshold for now.
            # threshold = np.percentile(loo_scores, 5) if loo_scores else -np.inf
            # Actually, let's use mean - 2*std of LOO scores.
            threshold = np.mean(loo_scores) - 2*np.std(loo_scores) if loo_scores else -np.inf
            
            current_sequence = [new_curve]
            current_feat = new_curve_feat
            total_log_prob = 0.0
            
            possible = True
            
            # Iterate backwards through the steps of the pattern
            # If pattern has 4 notes, steps are 0, 1, 2.
            # We start matching step 2 (N2 -> N3), then step 1 (N1 -> N2), etc.
            for step_idx in range(len(model_steps) - 1, -1, -1):
                step_model = model_steps[step_idx]
                
                best_prev_curve = None
                best_prev_prob = -np.inf
                
                # Search in recent curves
                # We only look at the last 'search_window' curves
                candidates = list(zip(valid_recent_curves, recent_feats))[-search_window:]
                # Iterate backwards through candidates
                for prev_curve, prev_feat in reversed(candidates):
                    # Skip curves that are already in the sequence (though unlikely with time constraints)
                    if prev_curve in current_sequence:
                        continue
                        
                    # Check if time makes sense (prev must be before current)
                    if prev_feat[0] >= current_feat[0]:
                        continue
                        
                    prob = self._calculate_step_log_prob(prev_feat, current_feat, step_model)
                    
                    if prob > best_prev_prob:
                        best_prev_prob = prob
                        best_prev_curve = prev_curve
                        best_prev_feat = prev_feat
                
                if best_prev_curve is not None and best_prev_prob > -np.inf:
                    current_sequence.insert(0, best_prev_curve)
                    current_feat = best_prev_feat
                    total_log_prob += best_prev_prob
                else:
                    possible = False
                    break
            
            if possible:
                # Normalize score? The user just said "comparably large".
                # LOO scores are sums of log probs.
                # print(f"Pattern {name} has total log prob {total_log_prob} and threshold {threshold}")
                if total_log_prob >= threshold:
                    detections.append({
                        'name': name,
                        'score': total_log_prob,
                        'curves': current_sequence,
                        'threshold': threshold
                    })
                    
        return detections
