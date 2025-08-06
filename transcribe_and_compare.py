import whisper
import pandas as pd
import os

# Load the whisper model (base, small, medium, etc.)
model = whisper.load_model("base")  # or use "small", "medium", "large"

# Load your transcript CSV
df = pd.read_csv("datasets/transcript/transcript.csv")

# Folder containing 16kHz mono .wav files
audio_folder = "datasets/audio"  # or /audio if already converted

results = []

for index, row in df.iterrows():
    audio_path = os.path.join(audio_folder, row["filename"])

    if not os.path.exists(audio_path):
        print(f" Missing file: {audio_path}")
        continue

    # Run transcription
    result = model.transcribe(audio_path, language="ne")

    predicted = result["text"].strip()
    actual = row["transcript"].strip()

    results.append({
        "filename": row["filename"],
        "actual": actual,
        "predicted": predicted
    })

# Save to CSV
pd.DataFrame(results).to_csv("whisper_transcription_results.csv", index=False)
print("Transcription complete. Results saved to whisper_transcription_results.csv")
