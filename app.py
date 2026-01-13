# app.py
# This is a simple Gradio-based app for Text-to-Speech (TTS) and Speech-to-Text (STT).
# It uses pyttsx3 for TTS (offline, pretrained voices via system engines like eSpeak or SAPI5).
# It uses speech_recognition with PocketSphinx for STT (offline, pretrained models).
# No API keys required. Run with: python app.py

import gradio as gr
import speech_recognition as sr
import pyttsx3
import tempfile
import os

# Initialize TTS engine
engine = pyttsx3.init()

def text_to_speech(text):
    # Save speech to a temporary audio file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        engine.save_to_file(text, tmp_file.name)
        engine.runAndWait()
        return tmp_file.name

def speech_to_text(audio):
    if audio is None:
        return "No audio input provided."
    
    r = sr.Recognizer()
    # Gradio audio input provides a file path
    with sr.AudioFile(audio) as source:
        audio_data = r.record(source)
    
    try:
        # Use PocketSphinx for offline recognition (pretrained model)
        text = r.recognize_sphinx(audio_data)
        return text
    except sr.UnknownValueError:
        return "Could not understand audio."
    except sr.RequestError as e:
        return f"Error with recognition: {e}"
    finally:
        # Clean up the temporary audio file from Gradio
        if os.path.exists(audio):
            os.remove(audio)

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Text-to-Speech and Speech-to-Text App")
    gr.Markdown("This app uses offline pretrained models. No API keys needed.")
    
    with gr.Tab("Text to Speech"):
        text_input = gr.Textbox(label="Enter text to convert to speech")
        audio_output = gr.Audio(label="Generated Speech", type="filepath")
        tts_button = gr.Button("Convert to Speech")
        tts_button.click(text_to_speech, inputs=text_input, outputs=audio_output)
    
    with gr.Tab("Speech to Text"):
        audio_input = gr.Audio(source="microphone", type="filepath", label="Record your speech")
        text_output = gr.Textbox(label="Recognized Text")
        stt_button = gr.Button("Convert to Text")
        stt_button.click(speech_to_text, inputs=audio_input, outputs=text_output)

demo.launch()
