import torchaudio
import whisper
import numpy as np
from speechbrain.inference.speaker import SpeakerRecognition
import librosa
import soundfile as sf
from pydub import AudioSegment
import os

# Load Whisper model
whisper_model = whisper.load_model("medium")

# Load SpeechBrain speaker recognition model
spk_model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmpdir")

# Convert M4A to WAV if necessary
AUDIO_FILE = "/home/nick_z/ADIRproject/adir-tool/.venv/transcriber/question50.m4a"  # Change to your M4A file

audio, sr = librosa.load(AUDIO_FILE, sr=16000)


# Transcribe using Whisper
print("Transcribing audio...")
transcription = whisper_model.transcribe(AUDIO_FILE, language="en")
print("\nFull Transcription:\n", transcription["text"])
segments = transcription["segments"]

# Speaker diarization function
def identify_speakers(segments, audio):
    speaker_labels = []
    for segment in segments:
        start, end = segment["start"], segment["end"]

        # Extract segment audio
        segment_audio = audio[int(start * sr):int(end * sr)]
        sf.write("temp.wav", segment_audio, sr)

        # Compare speaker embedding with previous segments
        if speaker_labels:
            prev_speaker = speaker_labels[-1]
            prev_audio, _ = librosa.load("temp_prev.wav", sr=16000)
            sf.write("temp_prev.wav", prev_audio, sr)

            # Compute similarity score
            score, _ = spk_model.verify_files("temp.wav", "temp_prev.wav")

            if score > 0.5:
                speaker_labels.append(prev_speaker)  # Same speaker
            else:
                speaker_labels.append(f"Speaker {len(set(speaker_labels)) + 1}")
        else:
            speaker_labels.append("Speaker 1")

        sf.write("temp_prev.wav", segment_audio, sr)  # Save for next comparison

    return speaker_labels

# Assign speakers to transcribed text
print("Performing speaker diarization...")
speaker_labels = identify_speakers(segments, audio)

# Display final result
for i, segment in enumerate(segments):
    print(f"{speaker_labels[i]}: {segment['text']}")

# Cleanup temporary files
os.remove("converted_audio.wav") if AUDIO_FILE == "converted_audio.wav" else None
os.remove("temp.wav") if os.path.exists("temp.wav") else None
os.remove("temp_prev.wav") if os.path.exists("temp_prev.wav") else None

print("Processing complete!")
