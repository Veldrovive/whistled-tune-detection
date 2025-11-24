import pyaudio
import numpy as np
import librosa
from collections import deque
from scipy.signal import find_peaks
import threading
import queue
import time

class Curve:
    def __init__(self, start_time, start_freq, start_amp, start_peak_index):
        # Points: list of (time, freq, amp)
        self.points = [(start_time, start_freq, start_amp)]
        self.last_peak_index = start_peak_index
        self.active = True

    def add_point(self, time, freq, amp, peak_index):
        self.points.append((time, freq, amp))
        self.last_peak_index = peak_index
        
    def __repr__(self):
        return f"Curve(len={len(self.points)}, start={self.points[0][0]:.2f}s, end={self.points[-1][0]:.2f}s, freq_start={self.points[0][1]:.1f}Hz, freq_end={self.points[-1][1]:.1f}Hz)"

class AudioStreamProcessorThreaded:
    def __init__(
        self,
        chunk_size=1024,
        rate=22050,
        n_fft=2048,
        hop_len=512,
        buffer_duration_s=5,
        n_mels=128,
        fmin=300,
        fmax=3500,
        max_curve_jump=2,
        min_interesting_curve_len=5,
        wav_filepath=None,
        device_name=None
    ):
        assert (chunk_size / hop_len).is_integer(), "Hop length must be an integer fraction of hop length"

        self.chunk_size = chunk_size
        self.rate = rate
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.buffer_duration_s = buffer_duration_s
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.wav_filepath = wav_filepath
        self.device_name = device_name
        self.hops_per_chunk = int(round(chunk_size / hop_len))
        self.max_curve_jump = max_curve_jump
        self.min_interesting_curve_len = min_interesting_curve_len

        self.mel_basis = librosa.filters.mel(
            sr=self.rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        self.mel_center_freqs = librosa.mel_frequencies(
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        self.hanning_window = np.hanning(self.n_fft)

        num_samples_in_buffer_duration = self.buffer_duration_s * self.rate
        self.num_buffer_samples = int(num_samples_in_buffer_duration / self.hop_len)
        assert self.num_buffer_samples > 0, "Number of buffer samples cannot be less that 0"

        self.spectrogram_buffer = deque(maxlen=self.num_buffer_samples)
        self.peaks_buffer = deque(maxlen=self.num_buffer_samples)
        self.tracked_curves = []
        self.finished_curves = []
        self.new_curves = []
        self.immediate_fft_buffer = np.zeros(n_fft, dtype=np.float32)

        self.total_samples_processed = 0
        
        self.p = None
        self.stream = None
        self.audio_data = None
        
        # Threading components
        self.result_queue = queue.Queue(maxsize=100) # Buffer for UI to consume
        self.running = False
        self.audio_thread = None

        if self.wav_filepath:
            print(f"Loading audio from file: {self.wav_filepath}")
            self.audio_data, _ = librosa.load(self.wav_filepath, sr=self.rate, mono=True)
        else:
            self._init_microphone(device_name)

    def _init_microphone(self, device_name):
        """Initializes PyAudio and the microphone stream."""
        print("Initializing microphone stream.")
        self.p = pyaudio.PyAudio()
        input_device_index = None

        if device_name:
            for i in range(self.p.get_device_count()):
                info = self.p.get_device_info_by_index(i)
                # Case-insensitive partial match for the device name
                if device_name.lower() in info['name'].lower() and info['maxInputChannels'] > 0:
                    input_device_index = i
                    print(f"Found matching device: '{info['name']}' at index {i}.")
                    break
            if input_device_index is None:
                print(f"Warning: Device '{device_name}' not found. Using default input device.")
                # Print the allowed devices
                print(f"Allowed device")
                for i in range(self.p.get_device_count()):
                    info = self.p.get_device_info_by_index(i)
                    print(f"\t{info['name']}")
        
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            input_device_index=input_device_index
        )
        print(f"Audio stream started at {self.rate} Hz from device index {input_device_index or 'default'}.")

    def _process_chunk(self, audio_chunk: np.ndarray):
        chunk_new_curves = []
        
        for hop_ind in range(self.hops_per_chunk):
            # Shift the immediate_fft_buffer by hop_len to make space
            self.immediate_fft_buffer[:-self.hop_len] = self.immediate_fft_buffer[self.hop_len:]
            # And then fill in the created space with the new audio
            self.immediate_fft_buffer[-self.hop_len:] = audio_chunk[hop_ind*self.hop_len:(hop_ind+1)*self.hop_len]

            # Now we can apply the fft
            windowed_chunk = self.immediate_fft_buffer * self.hanning_window
            stft_result = np.fft.rfft(windowed_chunk, n=self.n_fft)
            power_spectrum = np.abs(stft_result)**2
            mel_spectrum = np.dot(self.mel_basis, power_spectrum)
            mel_spectrogram_db = librosa.power_to_db(mel_spectrum, ref=1)
            self.spectrogram_buffer.append(mel_spectrogram_db)

            # We additionally compute the peaks of the signal for later processing
            peaks, properties = find_peaks(mel_spectrogram_db, width=4)
            self.peaks_buffer.append((peaks, properties))

            # Calculate current time for this frame
            current_time = (self.total_samples_processed + (hop_ind + 1) * self.hop_len - self.n_fft / 2) / self.rate
            
            # 1. Gather all possible connections (Greedy Matching Candidates)
            candidates = []
            for peak_index, peak in enumerate(peaks):
                if properties["prominences"][peak_index] < 25:
                    continue
                
                for curve_index, curve in enumerate(self.tracked_curves):
                    # curve.last_peak_index is the frequency index of the last point
                    distance = abs(peak - curve.last_peak_index)
                    if distance <= self.max_curve_jump:
                        candidates.append((distance, peak_index, curve_index))
            
            # 2. Sort by closest distance
            candidates.sort(key=lambda x: x[0])
            
            # 3. Assign strictly 1-to-1
            assigned_peaks = set()
            assigned_curves = set()
            
            # Create a new list for curves that continue to be tracked
            next_tracked_curves = []
            
            # Let's keep track of which curves from self.tracked_curves are continued
            continued_curve_indices = set()
            
            for dist, p_idx, c_idx in candidates:
                if p_idx not in assigned_peaks and c_idx not in assigned_curves:
                    # Commit the match
                    curve = self.tracked_curves[c_idx]
                    peak_val = peaks[p_idx]
                    freq = self.spec_y_to_freq(peak_val)
                    amp = properties["prominences"][p_idx] # Using prominence as amplitude proxy for now
                    
                    curve.add_point(current_time, freq, amp, peak_val)
                    
                    assigned_peaks.add(p_idx)
                    assigned_curves.add(c_idx)
                    continued_curve_indices.add(c_idx)
                    next_tracked_curves.append(curve)

            # 4. Handle Leftovers
            
            # Unassigned Peaks -> Start new curves
            for p_idx, peak in enumerate(peaks):
                if properties["prominences"][p_idx] < 25:
                    continue
                    
                if p_idx not in assigned_peaks:
                    freq = self.spec_y_to_freq(peak)
                    amp = properties["prominences"][p_idx]
                    new_curve = Curve(current_time, freq, amp, peak)
                    next_tracked_curves.append(new_curve)
            
            # Unassigned Curves -> Mark as finished
            for c_idx, curve in enumerate(self.tracked_curves):
                if c_idx not in continued_curve_indices:
                    if len(curve.points) >= self.min_interesting_curve_len:
                        # print(f"New finished interesting curve {curve}")
                        self.finished_curves.append(curve)
                        chunk_new_curves.append(curve)
            
            self.tracked_curves = next_tracked_curves
            
        self.total_samples_processed += self.chunk_size
        return chunk_new_curves

    def _audio_loop(self):
        """The main loop running in a separate thread."""
        print("Audio thread started.")
        while self.running:
            try:
                # Read from stream - BLOCKING call, but that's okay in this thread
                # exception_on_overflow=True to detect dropped frames
                data_bytes = self.stream.read(self.chunk_size, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data_bytes, dtype=np.float32)
                
                # Process the chunk
                new_curves = self._process_chunk(audio_chunk)
                
                # Send data to the main thread
                # We send a snapshot of the spectrogram buffer (last N frames) and new curves
                # We only send if the queue isn't full, to avoid blocking the audio thread
                if not self.result_queue.full():
                    # Convert deque to list for thread safety/pickling if needed, 
                    # though here we share memory. 
                    # We send the *latest* chunk's worth of spectrogram data.
                    # Actually, the main thread maintains the full buffer for display.
                    # We just need to send the NEW data.
                    
                    # Let's send the new spectrogram frames and new curves.
                    # The _process_chunk appends to self.spectrogram_buffer.
                    # We can just send the last hops_per_chunk frames.
                    
                    recent_spec = list(self.spectrogram_buffer)[-self.hops_per_chunk:]
                    recent_peaks = list(self.peaks_buffer)[-self.hops_per_chunk:]
                    
                    self.result_queue.put((recent_spec, recent_peaks, new_curves, self.total_samples_processed))
                else:
                    # If queue is full, we drop the UI update, but we KEEP processing audio!
                    # This is the key fix.
                    pass
                    
            except IOError as e:
                if not self.running:
                    break
                if e.errno == -9988: # Stream closed
                    print("\n[AUDIO THREAD] Stream closed, exiting loop.")
                    break
                print(f"\n[AUDIO THREAD ERROR] IO Error: {e}")
                continue
            except Exception as e:
                print(f"\n[AUDIO THREAD ERROR] Unexpected error: {e}")
                break
        print("Audio thread stopped.")

    def listen(self):
        """Starts the audio thread and yields results to the main thread."""
        self.running = True
        self.audio_thread = threading.Thread(target=self._audio_loop, daemon=True)
        self.audio_thread.start()
        
        try:
            while self.running:
                try:
                    # Wait for data from the audio thread
                    # timeout allows checking self.running periodically
                    data = self.result_queue.get(timeout=0.1)
                    yield data
                except queue.Empty:
                    continue
        finally:
            self.stop()

    def stop(self):
        self.running = False
        if self.audio_thread and self.audio_thread.is_alive():
            # We cannot join if we are called from the audio thread (unlikely here but good practice)
            if threading.current_thread() != self.audio_thread:
                self.audio_thread.join(timeout=1.0)
        self.close()

    def spec_y_to_freq(self, y_index: int) -> float:
        if not 0 <= y_index < self.n_mels:
            raise IndexError(f"y_index {y_index} is out of bounds for n_mels={self.n_mels}.")
        return self.mel_center_freqs[y_index]

    def close(self):
        if self.stream:
            print("Stopping audio stream.")
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                print(f"Error closing stream: {e}")
            self.stream = None
            
        if self.p:
            try:
                self.p.terminate()
            except Exception as e:
                print(f"Error terminating PyAudio: {e}")
            self.p = None
        else:
            print("Closing (no active stream to stop).")
