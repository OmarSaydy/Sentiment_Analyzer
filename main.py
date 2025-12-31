import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from st_audiorec import st_audiorec
import speech_recognition as sr
from pydub import AudioSegment
import io
import time

# ------------------ SETUP ------------------
st.set_page_config(page_title="Emotion AI", page_icon="🧠", layout="centered")

@st.cache_resource
def setup_nltk():
    nltk.download("punkt")
    nltk.download("vader_lexicon")
setup_nltk()

sia = SentimentIntensityAnalyzer()

# ------------------ FUNCTIONS ------------------
def analyze_sentiment(text):
    score = sia.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "positive", score
    elif score <= -0.05:
        return "negative", score
    else:
        return "neutral", score

def chatbot_response(sentiment):
    return {
        "positive": "😊 You sound happy! That’s wonderful.",
        "negative": "😟 You sound upset. I'm here for you.",
        "neutral": "😐 I understand. Tell me more."
    }[sentiment]

def typewriter(text):
    box = st.empty()
    shown = ""
    for char in text:
        shown += char
        box.markdown(f"### 🤖 {shown}")
        time.sleep(0.02)

def speech_to_text_web():
    audio_bytes = st_audiorec()
    if audio_bytes:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")
        audio.export("temp.wav", format="wav")

        r = sr.Recognizer()
        with sr.AudioFile("temp.wav") as source:
            audio_data = r.record(source)

        try:
            return r.recognize_google(audio_data)
        except:
            return "Could not understand audio."
    return ""

# ------------------ SESSION ------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ------------------ UI ------------------
st.title("🧠 Emotion-Aware AI Companion")
st.caption("Talk to me. I listen. I feel. I respond.")

tab1, tab2 = st.tabs(["💬 Text", "🎤 Voice"])

with tab1:
    user_text = st.text_input("Type something...")

with tab2:
    st.write("Press record and speak:")
    user_voice = speech_to_text_web()

user_input = user_text if user_text else user_voice

# ------------------ CHAT ENGINE ------------------
if user_input:
    sentiment, score = analyze_sentiment(user_input)
    response = chatbot_response(sentiment)
    st.session_state.history.append((user_input, sentiment, score, response))

# ------------------ DISPLAY ------------------
for msg, sentiment, score, response in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(msg)

    with st.chat_message("assistant"):
        st.markdown(f"{response}")
        st.progress((score + 1) / 2)

st.markdown("---")
st.caption("🚀 Multimodal Emotion AI · No PyAudio · Python 3.13 Ready")
