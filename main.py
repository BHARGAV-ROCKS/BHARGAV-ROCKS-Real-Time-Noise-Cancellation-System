import pyaudio
import wave
import numpy as np
import scipy.signal as signal
from noisereduce import reduce_noise
import webrtcvad
import time

# Define constants
CHUNK = 1024  # Audio chunk size
FORMAT = pyaudio.paInt16  # Format of audio input
CHANNELS = 1  # Mono audio
RATE = 16000  # Sampling rate
OUTPUT_FILE = "processed_audio.wav"

# Voice Activity Detection (VAD) setup
vad = webrtcvad.Vad(3)  # Aggressive mode for better silence detection


# Capture live audio stream from microphone
def record_audio(callback, duration=10):
    """
    Record audio from the microphone in real-time and process it using the provided callback function.

    :param callback: A function to process each chunk of audio.
    :param duration: Duration in seconds for recording.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                    input=True, frames_per_buffer=CHUNK)
    stream_output = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True)

    print("Recording...")
    frames = []

    for _ in range(0, int(RATE / CHUNK * duration)):
        start_time = time.time()
        data = stream.read(CHUNK)
        processed_data = callback(data)
        end_time = time.time()

        print(f"Processing latency: {(end_time - start_time) * 1000:.2f} ms")  # Log latency

        stream_output.write(processed_data)  # Play processed audio in real time
        frames.append(processed_data)

    print("Recording finished.")
    stream.stop_stream()
    stream.close()
    stream_output.stop_stream()
    stream_output.close()
    p.terminate()

    return frames, p


# Noise Reduction Functions
def dynamic_noise_profile(audio_data, rate):
    """
    Use VAD to detect silence and build a dynamic noise profile.
    """
    # Ensure the audio chunk matches VAD requirements
    frame_duration_ms = 10  # 10ms frame
    frame_length = int(rate * frame_duration_ms / 1000)  # 160 samples for 10ms at 16kHz

    if len(audio_data) < frame_length * 2:
        raise ValueError("Audio chunk is too short for VAD processing.")

    # Process each frame
    audio_array = np.frombuffer(audio_data, dtype=np.int16)

    # Check if the frame contains speech
    for start in range(0, len(audio_array), frame_length):
        end = start + frame_length
        frame = audio_array[start:end].tobytes()

        if len(frame) < frame_length * 2:
            # Skip incomplete frames
            continue

        if not vad.is_speech(frame, sample_rate=rate):
            # Silence detected, use this for noise reduction
            return reduce_noise(y=audio_array, sr=rate, prop_decrease=0.9)

    # Default: No silence detected, return the original audio
    return audio_array



def reduce_noise_single_speaker(audio_data, rate):
    """
    Enhanced single-speaker noise cancellation with adaptive profiling.
    """
    audio_array = np.frombuffer(audio_data, dtype=np.int16)

    # Apply dynamic noise profiling
    reduced_noise = dynamic_noise_profile(audio_data, rate)

    # Equalize to enhance voice frequencies (1kHz–3kHz)
    sos = signal.butter(10, [1000, 3000], btype='bandpass', fs=rate, output='sos')
    equalized_audio = signal.sosfilt(sos, reduced_noise)

    return equalized_audio.astype(np.int16).tobytes()


def reduce_noise_multiple_speakers(audio_data, rate):
    """
    Enhanced multi-speaker noise reduction with voice preservation.
    """
    audio_array = np.frombuffer(audio_data, dtype=np.int16)

    # Band-pass filter to focus on voice frequencies (300Hz–3400Hz)
    sos = signal.butter(10, [300, 3400], btype='bandpass', fs=rate, output='sos')
    filtered_audio = signal.sosfilt(sos, audio_array)

    # Apply noise reduction
    reduced_noise = reduce_noise(y=filtered_audio, sr=rate, prop_decrease=0.95)

    return reduced_noise.astype(np.int16).tobytes()


# Save processed audio to a .wav file
def save_audio(frames, p, file_name=OUTPUT_FILE):
    wf = wave.open(file_name, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    print(f"Processed audio saved as {file_name}")


# Main execution
if __name__ == "__main__":
    scenario = input("Choose scenario (single/multi): ").strip().lower()

    if scenario == "single":
        process_callback = lambda data: reduce_noise_single_speaker(data, RATE)
    elif scenario == "multi":
        process_callback = lambda data: reduce_noise_multiple_speakers(data, RATE)
    else:
        print("Invalid choice. Defaulting to single speaker mode.")
        process_callback = lambda data: reduce_noise_single_speaker(data, RATE)

    audio_frames, py_audio_instance = record_audio(callback=process_callback, duration=10)
    save_audio(audio_frames, py_audio_instance)
