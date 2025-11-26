import pyaudio
import numpy as np
import librosa
from collections import deque
from scipy.signal import find_peaks

import time
import logging

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

class AudioStreamProcessor:
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
        max_finished_curves=10,
        wav_filepath=None,
        device_name=None
    ):
        assert (chunk_size / hop_len).is_integer(), "Hop length must be an integer fraction of chunk size"
        self.wav_filepath = wav_filepath
        self.rate = rate
        self.chunk_size = chunk_size
        self.p = None
        self.stream = None
        self.audio_data = None
        if self.wav_filepath:
            assert self.rate is not None, "Rate must be specified if loading from a file"
            logging.info(f"Loading audio from file: {self.wav_filepath}")
            self.audio_data, _ = librosa.load(self.wav_filepath, sr=self.rate, mono=True)
        else:
            self._init_microphone(device_name)

        self.n_fft = n_fft
        self.hop_len = hop_len
        self.buffer_duration_s = buffer_duration_s
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.device_name = device_name
        self.hops_per_chunk = int(round(chunk_size / hop_len))
        self.max_curve_jump = max_curve_jump
        self.min_interesting_curve_len = min_interesting_curve_len
        self.max_finished_curves = max_finished_curves

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
        self.spectrogram_times = deque(maxlen=self.num_buffer_samples)
        self.peaks_buffer = deque(maxlen=self.num_buffer_samples)
        self.tracked_curves = []
        self.finished_curves = []
        self.new_curves = []
        self.immediate_fft_buffer = np.zeros(n_fft, dtype=np.float32)

        self.total_samples_processed = 0
        self.stream_start_time = None

    def _init_microphone(self, device_name):
        """Initializes PyAudio and the microphone stream."""
        logging.info("Initializing microphone stream.")
        self.p = pyaudio.PyAudio()
        input_device_index = None

        if device_name:
            for i in range(self.p.get_device_count()):
                info = self.p.get_device_info_by_index(i)
                # Case-insensitive partial match for the device name
                if device_name.lower() in info['name'].lower() and info['maxInputChannels'] > 0:
                    input_device_index = i
                    logging.info(f"Found matching device: '{info['name']}' at index {i}.")
                    break
            if input_device_index is None:
                logging.warning(f"Device '{device_name}' not found. Using default input device.")
                # Print the allowed devices
                logging.info(f"Allowed device")
                for i in range(self.p.get_device_count()):
                    info = self.p.get_device_info_by_index(i)
                    logging.info(f"\t{info['name']}")
        
        if self.rate is None:
            self.rate = int(self.p.get_device_info_by_index(input_device_index)['defaultSampleRate'])
        
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            input_device_index=input_device_index
        )
        logging.info(f"Audio stream started at {self.rate} Hz from device index {input_device_index or 'default'}.")
        self.input_device_index = input_device_index

    def restart_stream(self):
        """Restarts the audio stream."""
        logging.warning("Restarting audio stream...")
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            input_device_index=self.input_device_index
        )
        logging.info("Audio stream restarted.")

    def _read_chunk_with_timeout(self, timeout=5.0):
        """Reads a chunk from the stream with a timeout."""
        start_time = time.time()
        while self.stream.get_read_available() < self.chunk_size:
            if time.time() - start_time > timeout:
                raise TimeoutError("Audio read timed out")
            time.sleep(0.05)
        
        return self.stream.read(self.chunk_size, exception_on_overflow=False)

    def _process_chunk(self, audio_chunk: np.ndarray, chunk_start_time: float = None):
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
            # mel_spectrogram_db = librosa.power_to_db(mel_spectrum, ref=np.max)
            self.spectrogram_buffer.append(mel_spectrogram_db)

            # We additionally compute the peaks of the signal for later processing
            peaks, properties = find_peaks(mel_spectrogram_db, width=4)
            self.peaks_buffer.append((peaks, properties))

            # Calculate current time for this frame
            if chunk_start_time is not None:
                # Wall-clock mode: time is relative to the start of the chunk
                # The center of the frame is offset from the chunk start
                # The chunk start corresponds to the FIRST sample of the chunk.
                # The first hop (hop_ind=0) ends at hop_len.
                # Center of frame is at (hop_ind + 1) * hop_len - n_fft/2 relative to chunk start?
                # No, let's be precise.
                # chunk_start_time is the time of the first sample in audio_chunk.
                # The current frame uses samples from the buffer.
                # The buffer ends with the new data from this hop.
                # So the frame ends at chunk_start_time + (hop_ind + 1) * hop_len / rate
                
                frame_end_time = chunk_start_time + ((hop_ind + 1) * self.hop_len) / self.rate
                current_time = frame_end_time - (self.n_fft / 2) / self.rate
            else:
                # File mode: sample counting
                current_time = (self.total_samples_processed + (hop_ind + 1) * self.hop_len - self.n_fft / 2) / self.rate
            
            self.spectrogram_times.append(current_time)
            
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
                        logging.info(f"New finished interesting curve {curve}")
                        self.finished_curves.append(curve)
                        if len(self.finished_curves) > self.max_finished_curves:
                            self.finished_curves.pop(0)
                        self.new_curves.append(curve)
            
            self.tracked_curves = next_tracked_curves
            
            self.total_samples_processed += self.hop_len

    def listen(self):
        logging.info("Listen called...")
        if self.audio_data is not None:
            logging.info("Processing file...")
            # --- FILE MODE ---
            for i in range(0, len(self.audio_data), self.chunk_size):
                audio_chunk = self.audio_data[i:i + self.chunk_size]

                # If the last chunk is smaller than chunk_size, pad it with zeros
                if len(audio_chunk) < self.chunk_size:
                    audio_chunk = np.pad(audio_chunk, (0, self.chunk_size - len(audio_chunk)))

                self._process_chunk(audio_chunk, chunk_start_time=None)
                yield self.spectrogram_buffer, self.new_curves
                self.new_curves = []
            logging.info("Finished processing file")
        else:
            logging.info("Processing microphone...")
            # --- MICROPHONE MODE ---
            # c = 0
            while True:
                # logging.info("Microphone loop iteration...")
                try:
                    # if c % 1 == 0:
                    # logging.info("Reading audio chunk...")
                    # data_bytes = self.stream.read(self.chunk_size, exception_on_overflow=False)
                    data_bytes = self._read_chunk_with_timeout(timeout=2.0)
                    # logging.info(f"Read {len(data_bytes)} bytes")
                    audio_chunk = np.frombuffer(data_bytes, dtype=np.float32)
                    
                    # Calculate chunk start time based on wall clock
                    now = time.time()
                    if self.stream_start_time is None:
                        self.stream_start_time = now
                    
                    # We estimate the start of this chunk as (now - chunk_duration)
                    # relative to the stream start.
                    chunk_duration = self.chunk_size / self.rate
                    chunk_start_time = (now - self.stream_start_time) - chunk_duration
                    
                    self._process_chunk(audio_chunk, chunk_start_time=chunk_start_time)

                    # if c % 1 == 0:
                    # max_val = np.max(audio_chunk)
                    # min_val = np.min(audio_chunk)
                    # max_db_val = self.spectrogram_buffer[-1].max()
                    # min_db_val = self.spectrogram_buffer[-1].min()
                    # logging.info(f"Max: {max_val}, Min: {min_val}. Max DB: {max_db_val}, Min DB: {min_db_val}")
                    # c += 1
                    yield self.spectrogram_buffer, self.new_curves
                    self.new_curves = []
                except IOError as e:
                    logging.error(f"IO Error: {e}")
                    self.restart_stream()
                    continue
                except TimeoutError as e:
                    logging.error(f"Timeout Error: {e}")
                    self.restart_stream()
                    continue
                except Exception as e:
                    logging.error(f"Unexpected error in listen loop: {e}", exc_info=True)
                    self.restart_stream()
                    continue

    def spec_y_to_freq(self, y_index: int) -> float:
        """
        Converts a y-index (row) of the spectrogram to its corresponding frequency in Hz.

        Args:
            y_index (int): The row index in a spectrogram column (from 0 to n_mels-1).

        Returns:
            float: The center frequency of the Mel band in Hertz.
        
        Raises:
            IndexError: If the y_index is out of the valid range [0, n_mels-1].
        """
        if not 0 <= y_index < self.n_mels:
            raise IndexError(f"y_index {y_index} is out of bounds for n_mels={self.n_mels}.")
        return self.mel_center_freqs[y_index]

    def queue_index_to_sample(self, queue_index: int) -> int:
        """
        Calculates the absolute starting sample number for a spectrogram frame
        at a given index in the spectrogram_buffer (deque).

        Args:
            queue_index (int): The index within the spectrogram_buffer deque. 
                               Index 0 is the oldest frame, -1 is the newest.

        Returns:
            int: The absolute sample number in the original audio stream where the
                 FFT window for this spectrogram frame began.
        
        Raises:
            IndexError: If the queue_index is out of the current bounds of the buffer.
        """
        current_buffer_len = len(self.spectrogram_buffer)
        if not -current_buffer_len <= queue_index < current_buffer_len:
            raise IndexError(
                f"queue_index {queue_index} is out of bounds for the current buffer size of {current_buffer_len}."
            )
        
        # If a negative index is used (e.g., -1 for the last element), convert it to a positive one.
        if queue_index < 0:
            queue_index += current_buffer_len

        # The most recent frame's audio window *ended* at total_samples_processed.
        # Its window (of size n_fft) therefore *started* at total_samples_processed - n_fft.
        start_sample_of_newest_frame = self.total_samples_processed - self.n_fft
        
        # The frame at queue_index is a certain number of hops older than the newest frame.
        # The newest frame is at index (current_buffer_len - 1).
        num_hops_ago = (current_buffer_len - 1) - queue_index
        
        # The starting sample is the newest frame's start time, rewound by the number of hops.
        start_sample = start_sample_of_newest_frame - (num_hops_ago * self.hop_len)
        
        return int(start_sample)
        
    def close(self):
        if self.stream and self.p:
            logging.info("Stopping audio stream.")
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
        else:
            logging.info("Closing (no active stream to stop).")