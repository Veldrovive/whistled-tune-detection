import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from collections import Counter
from scipy.stats import norm
import math
from audio_lib_v2 import Curve

# Constants
PATTERN_DATA_DIR = Path(__file__).parent / "pattern_data"
PATTERN_MODELS_DIR = Path(__file__).parent / "pattern_models"
PATTERN_MODELS_DIR.mkdir(exist_ok=True)

def load_pattern_data(pattern_name):
    filename = PATTERN_DATA_DIR / f"{pattern_name}_pattern_data.pkl"
    if not filename.exists():
        raise FileNotFoundError(f"Pattern data file not found: {filename}")
    
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def extract_features(pattern):
    """
    Extracts onset time and centroid frequency for each curve in the pattern.
    Returns: list of (onset_time, centroid_freq)
    """
    features = []
    for curve in pattern:
        onset_time = curve.points[0][0]
        freqs = [p[1] for p in curve.points]
        centroid_freq = np.mean(freqs)
        features.append((onset_time, centroid_freq))
    return features

def compute_deltas(features):
    """
    Computes deltas between consecutive notes.
    Returns: list of (delta_time, delta_log_freq)
    """
    deltas = []
    for i in range(len(features) - 1):
        t1, f1 = features[i]
        t2, f2 = features[i+1]
        
        delta_time = t2 - t1
        delta_log_freq = math.log(f2) - math.log(f1)
        deltas.append((delta_time, delta_log_freq))
    return deltas

def fit_gaussians(all_deltas):
    """
    Fits Gaussians to the deltas at each step.
    all_deltas: list of list of deltas. Shape: (num_patterns, num_steps, 2)
    Returns: list of (mu_t, sigma_t, mu_f, sigma_f) for each step
    """
    num_steps = len(all_deltas[0])
    models = []
    
    for step in range(num_steps):
        # Gather data for this step
        time_deltas = [p[step][0] for p in all_deltas]
        freq_deltas = [p[step][1] for p in all_deltas]
        
        mu_t, sigma_t = norm.fit(time_deltas)
        mu_f, sigma_f = norm.fit(freq_deltas)
        
        models.append({
            'step': step,
            'time_dist': (mu_t, sigma_t),
            'freq_dist': (mu_f, sigma_f)
        })
        
    return models

def calculate_log_probability(deltas, model):
    """
    Calculates the log probability of a pattern given the model.
    """
    log_prob = 0.0
    for step, (dt, df) in enumerate(deltas):
        step_model = model[step]
        mu_t, sigma_t = step_model['time_dist']
        mu_f, sigma_f = step_model['freq_dist']
        
        # Add epsilon to sigma to avoid division by zero if variance is 0 (unlikely but safe)
        sigma_t = max(sigma_t, 1e-6)
        sigma_f = max(sigma_f, 1e-6)
        
        lp_t = norm.logpdf(dt, loc=mu_t, scale=sigma_t)
        lp_f = norm.logpdf(df, loc=mu_f, scale=sigma_f)
        
        log_prob += lp_t + lp_f
        
    return log_prob

def main():
    parser = argparse.ArgumentParser(description="Process whistled pattern data and train models.")
    parser.add_argument("pattern_name", type=str, help="Name of the pattern to process.")
    args = parser.parse_args()
    
    pattern_name = args.pattern_name
    print(f"Processing pattern: {pattern_name}")
    
    # 1. Load Data
    try:
        raw_patterns = load_pattern_data(pattern_name)
    except FileNotFoundError as e:
        print(e)
        return

    print(f"Loaded {len(raw_patterns)} raw patterns.")
    
    # 2. Filter by Length
    lengths = [len(p) for p in raw_patterns]
    if not lengths:
        print("No patterns found.")
        return
        
    mode_length = Counter(lengths).most_common(1)[0][0]
    print(f"Mode pattern length: {mode_length}")
    
    valid_patterns = [p for p in raw_patterns if len(p) == mode_length]
    rejected_count = len(raw_patterns) - len(valid_patterns)
    if rejected_count > 0:
        print(f"Warning: Rejected {rejected_count} patterns with different lengths.")
    
    if len(valid_patterns) < 2:
        print("Error: Need at least 2 valid patterns to perform LOO evaluation.")
        return

    # 3. Extract Features and Deltas
    all_pattern_deltas = []
    for p in valid_patterns:
        feats = extract_features(p)
        deltas = compute_deltas(feats)
        all_pattern_deltas.append(deltas)
        
    # 4. Leave-One-Out Evaluation
    loo_scores = []
    
    for i in range(len(all_pattern_deltas)):
        # Train on all except i
        train_deltas = all_pattern_deltas[:i] + all_pattern_deltas[i+1:]
        test_deltas = all_pattern_deltas[i]
        
        model = fit_gaussians(train_deltas)
        score = calculate_log_probability(test_deltas, model)
        loo_scores.append(score)
        
    print(f"LOO Scores: Mean={np.mean(loo_scores):.2f}, Std={np.std(loo_scores):.2f}")
    
    # 5. Train Final Model on All Data
    final_model = fit_gaussians(all_pattern_deltas)
    
    # Save Model
    model_filename = PATTERN_MODELS_DIR / f"{pattern_name}_model.pkl"
    save_data = {
        'model': final_model,
        'loo_scores': loo_scores,
        'pattern_length': mode_length
    }
    with open(model_filename, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"Saved model to {model_filename}")
    
    # 6. Visualization
    num_steps = mode_length - 1
    
    # Figure 1: Distributions
    fig1, axes = plt.subplots(num_steps, 2, figsize=(12, 4 * num_steps))
    if num_steps == 1:
        axes = np.expand_dims(axes, 0)
        
    for step in range(num_steps):
        # Time Deltas
        time_deltas = [p[step][0] for p in all_pattern_deltas]
        mu_t, sigma_t = final_model[step]['time_dist']
        
        ax_t = axes[step, 0]
        ax_t.hist(time_deltas, bins=10, density=True, alpha=0.6, color='b')
        xmin, xmax = ax_t.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu_t, sigma_t)
        ax_t.plot(x, p, 'k', linewidth=2)
        ax_t.set_title(f"Step {step+1}: Time Delta (mu={mu_t:.2f}, sigma={sigma_t:.2f})")
        
        # Freq Deltas
        freq_deltas = [p[step][1] for p in all_pattern_deltas]
        mu_f, sigma_f = final_model[step]['freq_dist']
        
        ax_f = axes[step, 1]
        ax_f.hist(freq_deltas, bins=10, density=True, alpha=0.6, color='r')
        xmin, xmax = ax_f.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu_f, sigma_f)
        ax_f.plot(x, p, 'k', linewidth=2)
        ax_f.set_title(f"Step {step+1}: Log Freq Delta (mu={mu_f:.2f}, sigma={sigma_f:.2f})")
        
    plt.tight_layout()
    plt.show()
    
    # Figure 2: LOO Scores
    plt.figure(figsize=(8, 6))
    plt.hist(loo_scores, bins=10, color='g', alpha=0.7)
    plt.title(f"Leave-One-Out Log-Likelihood Scores (Mean={np.mean(loo_scores):.2f})")
    plt.xlabel("Log Probability")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.show()

    # Figure 3: Relative Note Distribution
    plt.figure(figsize=(10, 8))
    
    # Prepare data for scatter plot
    all_rel_times = []
    all_rel_freqs = []
    all_scores = []
    note_indices = []
    
    for i, pattern in enumerate(valid_patterns):
        score = loo_scores[i]
        features = extract_features(pattern)
        
        # Reference note (first note)
        t0, f0 = features[0]
        log_f0 = math.log(f0)
        
        for j, (t, f) in enumerate(features):
            rel_time = t - t0
            rel_log_freq = math.log(f) - log_f0
            
            all_rel_times.append(rel_time)
            all_rel_freqs.append(rel_log_freq)
            all_scores.append(score)
            note_indices.append(j)
            
    # Scatter plot
    sc = plt.scatter(all_rel_times, all_rel_freqs, c=all_scores, cmap='viridis', alpha=0.6, s=50)
    plt.colorbar(sc, label='Pattern Log Probability')
    
    # Annotate clusters (optional, but helpful to see note order)
    # We can plot text numbers at the mean location of each note index
    for j in range(mode_length):
        # Filter points for this note index
        indices = [k for k, idx in enumerate(note_indices) if idx == j]
        times = [all_rel_times[k] for k in indices]
        freqs = [all_rel_freqs[k] for k in indices]
        
        mean_t = np.mean(times)
        mean_f = np.mean(freqs)
        
        plt.text(mean_t, mean_f, str(j+1), fontsize=12, fontweight='bold', color='black', 
                 ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    plt.title(f"Relative Note Distribution (Aligned to Note 1)")
    plt.xlabel("Relative Time (s)")
    plt.ylabel("Relative Log Frequency")
    plt.grid(True, alpha=0.3)
    plt.show()

    # Figure 4: Relative Note Distribution (Outliers Removed)
    plt.figure(figsize=(10, 8))
    
    # Filter outliers (keep top 90%)
    threshold = np.percentile(loo_scores, 10)
    print(f"Filtering outliers below score: {threshold:.2f}")
    
    filtered_rel_times = []
    filtered_rel_freqs = []
    filtered_scores = []
    filtered_note_indices = []
    
    for i, pattern in enumerate(valid_patterns):
        score = loo_scores[i]
        if score < threshold:
            continue
            
        features = extract_features(pattern)
        
        # Reference note (first note)
        t0, f0 = features[0]
        log_f0 = math.log(f0)
        
        for j, (t, f) in enumerate(features):
            rel_time = t - t0
            rel_log_freq = math.log(f) - log_f0
            
            filtered_rel_times.append(rel_time)
            filtered_rel_freqs.append(rel_log_freq)
            filtered_scores.append(score)
            filtered_note_indices.append(j)
            
    # Scatter plot
    sc = plt.scatter(filtered_rel_times, filtered_rel_freqs, c=filtered_scores, cmap='viridis', alpha=0.6, s=50)
    plt.colorbar(sc, label='Pattern Log Probability')
    
    # Annotate clusters
    for j in range(mode_length):
        indices = [k for k, idx in enumerate(filtered_note_indices) if idx == j]
        if indices:
            times = [filtered_rel_times[k] for k in indices]
            freqs = [filtered_rel_freqs[k] for k in indices]
            
            mean_t = np.mean(times)
            mean_f = np.mean(freqs)
            
            plt.text(mean_t, mean_f, str(j+1), fontsize=12, fontweight='bold', color='black', 
                     ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    plt.title(f"Relative Note Distribution (Top 90% by Probability)")
    plt.xlabel("Relative Time (s)")
    plt.ylabel("Relative Log Frequency")
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()
