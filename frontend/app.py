import os
import base64
import tempfile
import streamlit as st
import re
from indic_transliteration.sanscript import transliterate, ITRANS, DEVANAGARI

st.set_page_config(page_title="Bilingual Subtitle Generator", page_icon="📝", layout="wide")

st.title("Bilingual Subtitle Generator")
st.write("Upload a video file. The app will generate subtitles in Nepali (देवनागरी) and English, and show them while you play the video.")

uploaded = st.file_uploader("Upload your video", type=["mp4"], accept_multiple_files=False)

NEPALI_COMMON = {
    "mero", "timro", "timi", "ma", "tapai", "bhaneko", "vayena", "cha", "xa", "bhayo", "huncha", "parcha", "garna", "lagxa", "ramro", "sathi", "namaste", "dhanyabad", "ghar", "bholi", "aaja", "hajur"
    # Add more Nepali words as needed
}

def is_nepali_word(word):
    return word.lower() in NEPALI_COMMON or word.lower().endswith(("cha", "xa", "nu", "ko", "le", "la", "na", "yo", "re", "bhayo", "vayena", "lagxa", "parxa", "garnu", "garna"))

def roman_to_devanagari(word):
    try:
        return transliterate(word, ITRANS, DEVANAGARI)
    except Exception:
        return word

def mix_bilingual_line(text):
    tokens = re.findall(r"([\w\-']+|[^\w\s])", text)
    out = []
    for t in tokens:
        if is_nepali_word(t):
            out.append(roman_to_devanagari(t))
        else:
            out.append(t)
    return "".join(out)

def to_srt_time(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def build_webvtt(segments: list) -> str:
    lines = ["WEBVTT", ""]
    for seg in segments:
        start = to_srt_time(seg["start"]).replace(",", ".")
        end = to_srt_time(seg["end"]).replace(",", ".")
        lines.append(f"{start} --> {end}")
        lines.append(seg["bilingual"])
        lines.append("")
    return "\n".join(lines)

def merge_segments(segments: list, max_chars: int = 80) -> list:
    merged = []
    cur = None
    for seg in segments:
        s, e, t = float(seg["start"]), float(seg["end"]), str(seg["text"]).strip()
        if not t:
            continue
        if cur is None:
            cur = {"start": s, "end": e, "text": t}
        else:
            if len(cur["text"]) + 1 + len(t) <= max_chars:
                cur["text"] += (" " + t)
                cur["end"] = e
            else:
                merged.append(cur)
                cur = {"start": s, "end": e, "text": t}
    if cur is not None:
        merged.append(cur)
    return merged

if uploaded:
    suffix = os.path.splitext(uploaded.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = tmp.name

    try:
        import whisper
        st.info("Transcribing your video. Please wait...")
        model = whisper.load_model("medium")
        result = model.transcribe(tmp_path, language="en")  # <--- IMPORTANT: use "en"
        raw_segments = result.get("segments", [])
        segments = merge_segments(raw_segments, max_chars=80)

        # Post-process for bilingual output
        for ch in segments:
            ch["bilingual"] = mix_bilingual_line(ch["text"])

        st.subheader("Subtitles")
        for ch in segments:
            st.markdown(f"{ch['bilingual']}\n---")

        vtt_text = build_webvtt(segments)
        vtt_b64 = base64.b64encode(vtt_text.encode("utf-8")).decode("ascii")
        with open(tmp_path, "rb") as f:
            media_bytes = f.read()
        media_b64 = base64.b64encode(media_bytes).decode("ascii")
        media_mime = "video/mp4"

        st.subheader("Video with Real-Time Subtitles")
        html = f"""
        <video controls style="max-width:100%" crossorigin="anonymous">
          <source src="data:{media_mime};base64,{media_b64}">
          <track label="Bilingual" kind="subtitles" srclang="ne" default src="data:text/vtt;base64,{vtt_b64}">
        </video>
        """
        st.components.v1.html(html, height=420)

    except Exception as e:
        st.error("Sorry, something went wrong while processing your video.")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass