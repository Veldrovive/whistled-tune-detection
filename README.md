# Whistle House

Whistle House is a Python-based system that allows you to trigger events (like turning on smart lights) by whistling specific melodies. It uses real-time audio processing to detect pure tones (whistles), tracks them as "curves" (notes), and matches sequences of these curves against trained statistical models.

## Theory of Operation

The system operates in a pipeline:

1.  **Audio Input & Spectrogram**:
    *   Audio is captured via `PyAudio`.
    *   A Mel Spectrogram is computed using `librosa` (Short-Time Fourier Transform).
    *   This converts the audio signal into a time-frequency representation.

2.  **Curve Detection (Note Tracking)**:
    *   We detect peaks in the spectrogram for each time frame.
    *   Peaks are connected across frames to form **Curves**.
    *   A "Curve" represents a continuous whistle note. It tracks the frequency trajectory over time.
    *   Whistling produces a strong fundamental frequency (pure tone), making this approach robust compared to voice or humming.

3.  **Pattern Recognition**:
    *   **Features**: For each curve, we extract the **Onset Time** and **Centroid Frequency**.
    *   **Deltas**: We model patterns not by absolute notes, but by the *transitions* between them. We calculate:
        *   $\Delta t$: Time difference between consecutive notes.
        *   $\Delta \log(f)$: Log-frequency difference (interval) between consecutive notes.
    *   **Statistical Model**: For each step in a pattern (e.g., Note 1 $\to$ Note 2), we fit Gaussian distributions to these deltas based on training data.
    *   **Matching**: When a new note finishes, the system searches backwards through recent notes to find a sequence that maximizes the log-probability of the transitions according to the trained model.

## Installation

Ensure you have Python 3.10+ installed.

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Veldrovive/whistled-tune-detection
    cd whistled-tune-detection
    ```

2.  **Install Dependencies**:
    ```bash
    pip install pyaudio numpy scipy matplotlib librosa
    ```

## Usage

### 1. Collect Training Data

To define a new pattern (e.g., "rising"), you need to record examples of it.

```bash
python collect_pattern_data.py rising
```
*   **Instructions**: Whistle your pattern clearly. Pause for about 1 second between repetitions.
*   The script will visualize the spectrogram and detected notes in real-time.
*   Close the window or press Ctrl+C to save the data to `pattern_data/rising_pattern_data.pkl`.

### 2. Train the Model

Once you have collected data, process it to create a statistical model.

```bash
python process_pattern_data.py rising
```
*   This script filters outliers (patterns of incorrect length).
*   It trains Gaussian models on the transitions.
*   It performs a Leave-One-Out (LOO) evaluation to determine a good detection threshold.
*   The model is saved to `pattern_models/rising_model.pkl`.
*   It also displays plots showing the distribution of your notes.

### 3. Run the Listener

You can now run a script that listens for these patterns.

**Live Visualization Test:**
```bash
python test_pattern_detection_live.py
```
This will show a live plot and print to the console when patterns are detected.

**Example Application (Kasa Lights):**
`kasa_light_demo.py` demonstrates how to control TP-Link Kasa smart plugs/bulbs.
```bash
python kasa_light_demo.py
```
*   Requires `python-kasa`.
*   Edit the script to match your device IPs or MAC addresses.

## Project Structure

*   `audio_lib.py`: Core audio processing engine. Handles microphone input, spectrogram generation, and curve tracking.
*   `collect_pattern_data.py`: Tool for recording training data.
*   `process_pattern_data.py`: Tool for training models from recorded data.
*   `pattern_recognition.py`: Logic for matching live audio against trained models.
*   `pattern_event_handler.py`: High-level class (`PatternEventListener`) for easy integration into apps.
*   `kasa_light_demo.py`: Example application.