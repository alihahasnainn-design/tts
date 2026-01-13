import streamlit as st
import whisper
from gtts import gTTS
import os
import torch

# This helps the app run on CPU-only cloud servers
device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_whisper():
    # Load the smallest model to prevent memory crashes
    return whisper.load_model("tiny", device=device)

st.set_page_config(page_title="TTS & STT Tool")
st.title("🎙️ AI Audio Converter")

try:
    model = load_whisper()
    st.success("AI Model Loaded Successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# --- Speech to Text ---
st.subheader("1. Audio to Text")
audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a"])

if audio_file and st.button("Transcribe"):
    with st.spinner("Transcribing..."):
        with open("temp_audio.mp3", "wb") as f:
            f.write(audio_file.read())
        
        result = model.transcribe("temp_audio.mp3")
        st.write("**Transcription:**")
        st.info(result["text"])
        os.remove("temp_audio.mp3")

# --- Text to Speech ---
st.subheader("2. Text to Audio")
user_text = st.text_input("Type something here...")

if user_text and st.button("Convert to Voice"):
    tts = gTTS(text=user_text, lang='en')
    tts.save("speech.mp3")
    st.audio("speech.mp3")
