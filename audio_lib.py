import pyaudio
import numpy as np
import librosa
from collections import deque
from swift_f0 import SwiftF0

class AudioStreamProcessor:
    """
    A class to handle audio processing from a live stream or a pre-recorded file.
    It captures or loads audio, computes spectrograms and detects pitch,
    and maintains rolling buffers of this data.
    """
    def __init__(self,
                 chunk_size=2048,
                 rate=44100,
                 buffer_duration_s=5,
                 pitch_detector='yin',
                 wav_filepath=None,
                 device_name=None):
        """
        Initializes the audio stream/file and buffers.

        :param chunk_size: Number of audio samples per frame.
        :param rate: Sampling rate in Hz.
        :param buffer_duration_s: How many seconds of data to keep in the buffers.
        :param pitch_detector: The algorithm to use ('yin' or 'swiftf0').
        :param wav_filepath: Path to a WAV/audio file to process instead of the mic.
        :param device_name: Substring of the name of the audio device to use (e.g., "MacBook Pro Microphone").
                            If None, the default input device is used.
        """
        self.CHUNK = chunk_size
        self.RATE = rate
        self.BUFFER_DURATION_S = buffer_duration_s
        self.pitch_detector_type = pitch_detector
        self.wav_filepath = wav_filepath
        self.audio_data = None # To store loaded audio file data

        # --- Pitch Detector Initialization ---
        self.swift_detector = None
        self.SWIFT_TARGET_RATE = 16000 # SwiftF0 requires 16kHz audio
        if self.pitch_detector_type == 'swiftf0':
            print("Using SwiftF0 pitch detector.")
            self.swift_detector = SwiftF0(confidence_threshold=0.85)
        elif self.pitch_detector_type == 'yin':
            print("Using YIN pitch detector.")
        else:
            raise ValueError("Unsupported pitch_detector. Choose 'yin' or 'swiftf0'.")

        self.NUM_BUFFER_CHUNKS = int(self.BUFFER_DURATION_S * self.RATE / self.CHUNK)
        self.audio_buffer = deque(maxlen=self.NUM_BUFFER_CHUNKS)
        self.spectrogram_buffer = deque(maxlen=self.NUM_BUFFER_CHUNKS)
        self.pitch_buffer = deque(maxlen=self.NUM_BUFFER_CHUNKS)

        # --- Conditional Initialization: File or Microphone ---
        self.p = None
        self.stream = None
        if self.wav_filepath:
            print(f"Loading audio from file: {self.wav_filepath}")
            # Load the file, ensuring it's mono and at the target sample rate
            self.audio_data, _ = librosa.load(self.wav_filepath, sr=self.RATE, mono=True)
            print("File loaded successfully.")
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
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
            input_device_index=input_device_index
        )
        print(f"Audio stream started at {self.RATE} Hz from device index {input_device_index or 'default'}.")

    @staticmethod
    def list_audio_devices():
        """
        A helper static method to list available input audio devices.
        """
        p = pyaudio.PyAudio()
        print("Available Input Audio Devices:")
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"  - Index {i}: {info['name']}")
        p.terminate()

    def _process_chunk(self, audio_chunk: np.ndarray):
        """Processes a single chunk of audio data."""
        # --- Spectrogram Calculation (Unaffected by pitch detector choice) ---
        windowed_chunk = audio_chunk * np.hanning(len(audio_chunk))
        fft_complex = np.fft.rfft(windowed_chunk)
        fft_mag = np.abs(fft_complex)
        fft_db = 20 * np.log10(fft_mag + 1e-6)
        self.spectrogram_buffer.append(fft_db)

        # --- Pitch Detection (Swappable implementation) ---
        pitch_hz = 0.0  # Default to unvoiced
        if self.pitch_detector_type == 'swiftf0':
            audio_16k = librosa.resample(y=audio_chunk, orig_sr=self.RATE, target_sr=self.SWIFT_TARGET_RATE)
            result = self.swift_detector.detect_from_array(audio_16k, self.SWIFT_TARGET_RATE)
            voiced_pitches = result.pitch_hz[result.voicing]
            if len(voiced_pitches) > 0:
                pitch_hz = np.median(voiced_pitches)
        else: # 'yin'
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio_chunk,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C6'),
                sr=self.RATE
            )
            if not np.all(np.isnan(f0)):
                pitch_hz = float(np.nanmedian(f0)) # Ensure it's a standard float

        self.pitch_buffer.append(pitch_hz)

    def listen(self):
        """
        A generator that yields data buffers from the microphone OR a file.
        """
        if self.audio_data is not None:
            # --- FILE MODE ---
            print(f"Processing file '{self.wav_filepath}' in chunks...")
            for i in range(0, len(self.audio_data), self.CHUNK):
                audio_chunk = self.audio_data[i:i + self.CHUNK]
                
                # If the last chunk is smaller than CHUNK, pad it with zeros
                if len(audio_chunk) < self.CHUNK:
                    audio_chunk = np.pad(audio_chunk, (0, self.CHUNK - len(audio_chunk)))

                self.audio_buffer.append(audio_chunk)
                self._process_chunk(audio_chunk)
                yield self.spectrogram_buffer, self.pitch_buffer
            print("Finished processing file.")
        else:
            # --- MICROPHONE MODE ---
            print("Listening... Press Ctrl+C to stop.")
            while True:
                try:
                    data_bytes = self.stream.read(self.CHUNK, exception_on_overflow=False)
                    audio_chunk = np.frombuffer(data_bytes, dtype=np.float32)
                    self.audio_buffer.append(audio_chunk)
                    self._process_chunk(audio_chunk)
                    yield self.spectrogram_buffer, self.pitch_buffer
                except IOError as e:
                    print(f"IO Error: {e}")
                    continue

    def close(self):
        """Closes the audio stream gracefully if it was opened."""
        if self.stream and self.p:
            print("Stopping audio stream.")
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
        else:
            print("Closing (no active stream to stop).")