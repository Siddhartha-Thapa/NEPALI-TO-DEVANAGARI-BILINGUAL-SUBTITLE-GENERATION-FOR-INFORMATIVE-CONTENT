import whisper
import pandas as pd
import os
import fasttext
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# ---------- CONFIG ----------
AUDIO_FOLDER = "datasets/audio"
CSV_PATH = "datasets/transcript/transcript.csv"
OUTPUT_CSV = "bilingual_subtitles.csv"
FASTTEXT_MODEL_PATH = "lid.176.bin"  # Pre-trained fastText language ID model
# ----------------------------

# Load fastText language identification model
if not os.path.exists(FASTTEXT_MODEL_PATH):
    import urllib.request
    print("Downloading fastText language ID model...")
    urllib.request.urlretrieve(
        "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
        FASTTEXT_MODEL_PATH
    )
lang_model = fasttext.load_model(FASTTEXT_MODEL_PATH)

# Load Whisper model (larger = more accurate)
model = whisper.load_model("medium")  # try "large" if GPU/CPU can handle it

# Load transcript CSV (only used for comparison if you want accuracy check)
df = pd.read_csv(CSV_PATH)

results = []

for _, row in df.iterrows():
    audio_path = os.path.join(AUDIO_FOLDER, row["filename"])

    if not os.path.exists(audio_path):
        print(f"❌ Missing file: {audio_path}")
        continue

    # Force Whisper to treat input as English phonetics
    result = model.transcribe(audio_path, language="en")
    romanized_text = result["text"].strip()

    # Language ID + Transliteration
    bilingual_words = []
    for word in romanized_text.split():
        lang_pred = lang_model.predict(word.lower())[0][0]  # e.g., '__label__en' or '__label__ne'
        if "ne" in lang_pred:  # Nepali detected
            devanagari_word = transliterate(word, sanscript.ITRANS, sanscript.DEVANAGARI)
            bilingual_words.append(devanagari_word)
        else:
            bilingual_words.append(word)

    bilingual_text = " ".join(bilingual_words)

    results.append({
        "filename": row["filename"],
        "romanized_subtitle": romanized_text,
        "bilingual_subtitle": bilingual_text
    })

# Save results
pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
print(f"✅ Done! Bilingual subtitles saved to {OUTPUT_CSV}")
