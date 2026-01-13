import streamlit as st
import whisper
import pyttsx3
import os
from pydub import AudioSegment

# Initialize TTS engine
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Load Whisper Model (Pre-trained)
@st.cache_resource
def load_whisper():
    return whisper.load_model("base") # 'base' is a good balance of speed/accuracy

model = load_whisper()

st.title("🎙️ Offline STT & TTS App")

# --- SECTION 1: Speech to Text ---
st.header("Speech to Text")
audio_file = st.file_uploader("Upload an audio file (wav, mp3, m4a)", type=["wav", "mp3", "m4a"])

if audio_file is not None:
    if st.button("Transcribe"):
        with st.spinner("Transcribing..."):
            # Save temporary file
            with open("temp_audio", "wb") as f:
                f.write(audio_file.read())
            
            result = model.transcribe("temp_audio")
            st.success("Transcription Complete!")
            st.text_area("Result:", result["text"], height=150)
            os.remove("temp_audio")

# --- SECTION 2: Text to Speech ---
st.header("Text to Speech")
user_text = st.text_input("Enter text to convert to speech:")

if st.button("Play Audio"):
    if user_text:
        with st.spinner("Generating Speech..."):
            speak_text(user_text)
    else:
        st.warning("Please enter some text first.")
