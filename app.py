import streamlit as st
import whisper
from gtts import gTTS
import os

# Load Whisper Model (Pre-trained)
@st.cache_resource
def load_whisper():
    return whisper.load_model("tiny") # Using 'tiny' for faster performance on servers

model = load_whisper()

st.title("🎙️ Offline-Style STT & TTS App")

# --- SECTION 1: Speech to Text (STT) ---
st.header("Speech to Text")
audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])

if audio_file is not None:
    if st.button("Transcribe"):
        with st.spinner("Whisper is thinking..."):
            with open("temp_audio", "wb") as f:
                f.write(audio_file.read())
            
            result = model.transcribe("temp_audio")
            st.success("Transcription Complete!")
            st.text_area("Transcribed Text:", result["text"], height=150)
            os.remove("temp_audio")

# --- SECTION 2: Text to Speech (TTS) ---
st.header("Text to Speech")
user_text = st.text_input("Enter text to convert to speech:")

if st.button("Generate Audio"):
    if user_text:
        with st.spinner("Converting text to speech..."):
            # Create speech object
            tts = gTTS(text=user_text, lang='en')
            tts.save("speech.mp3")
            
            # Play in Streamlit
            audio_file = open("speech.mp3", 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/mp3')
            audio_file.close()
    else:
        st.warning("Please enter some text.")
