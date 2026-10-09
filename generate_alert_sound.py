"""
Generate Emergency Alert Sound

This script generates a simple alert sound file.
Run this script once to generate the alert.mp3 file.

Requirements: pip install pydub
"""

import struct
import wave
import math
import os

def generate_alert_sound(filename="alerts/emergency_alert.mp3", duration=1.0, frequency=880, sample_rate=44100):
    """Generate a simple alert beep sound."""
    # Create alerts directory
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Generate WAV first (simpler)
    wav_filename = filename.replace('.mp3', '.wav')

    num_samples = int(duration * sample_rate)

    with wave.open(wav_filename, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)

        for i in range(num_samples):
            t = i / sample_rate
            # Sine wave with envelope
            envelope = 1.0
            if i < num_samples * 0.1:  # Attack
                envelope = i / (num_samples * 0.1)
            elif i > num_samples * 0.8:  # Release
                envelope = (num_samples - i) / (num_samples * 0.2)

            # Frequency modulation for urgency
            freq_mod = 1 + 0.1 * math.sin(2 * math.pi * 5 * t)
            sample = envelope * math.sin(2 * math.pi * frequency * freq_mod * t)
            sample = int(sample * 32767 * 0.8)  # Volume

            wav_file.writeframes(struct.pack('h', sample))

    print(f"Generated: {wav_filename}")

    # Try to convert to MP3 if pydub available
    try:
        from pydub import AudioSegment
        sound = AudioSegment.from_wav(wav_filename)
        sound.export(filename, format="mp3")
        print(f"Converted to: {filename}")
        os.remove(wav_filename)
    except ImportError:
        print("pydub not installed, using WAV file instead")
        print(f"Rename {wav_filename} to use as alert sound")

    return filename

if __name__ == "__main__":
    generate_alert_sound()
