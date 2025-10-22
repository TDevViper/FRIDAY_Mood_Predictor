import streamlit as st
from transformers import pipeline
import emoji

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="FRIDAY Mood Predictor",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 FRIDAY Mood Predictor (Text Edition)")
st.write("Type your message or command below and FRIDAY will detect your mood!")

# -------------------------------
# Initialize Emotion Model
# -------------------------------
@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="bhadresh-savani/bert-base-go-emotion",
        return_all_scores=True
    )

emotion_analyzer = load_model()

# -------------------------------
# Preprocess Emojis
# -------------------------------
def preprocess_text(text):
    text = emoji.demojize(text)
    text = text.replace("_", " ").replace(":", " ")
    return text.strip()

# -------------------------------
# Analyze Emotion
# -------------------------------
def analyze_emotion(text):
    clean_text = preprocess_text(text)
    results = emotion_analyzer(clean_text)[0]
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
    return sorted_results[:3]  # top 3 emotions

# -------------------------------
# FRIDAY Response
# -------------------------------
def friday_reply(top_emotions):
    main_emotion = top_emotions[0]['label']
    confidence = top_emotions[0]['score']

    responses = {
        'joy': "You sound happy today 😄! Let’s keep the energy up!",
        'sadness': "You seem a bit down 😢. Want me to play something relaxing?",
        'anger': "Whoa, you sound upset 😤. Let’s calm down together.",
        'love': "Aww ❤️ love is in the air!",
        'excitement': "That’s awesome 🔥 You’re pumped up!",
        'fear': "You sound worried 😟 — wanna talk about it?",
        'neutral': "Alright, staying balanced as always 😌.",
        'optimism': "I like your positive vibe 🌞!",
        'remorse': "Hmm… sounds like regret 😔 — wanna move past it?",
        'nervousness': "Feeling anxious 😬? I got you, we’ll handle it."
    }

    reply = responses.get(main_emotion, f"I sense {main_emotion} 🧠")
    return f"Detected Emotion: {main_emotion.capitalize()} ({confidence:.2f})\nFRIDAY: {reply}"

# -------------------------------
# Streamlit Input
# -------------------------------
user_input = st.text_input("Type your message here:")

if st.button("Analyze Mood") and user_input:
    top_emotions = analyze_emotion(user_input)
    st.markdown("### Top Emotions")
    for e in top_emotions:
        st.write(f"{e['label'].capitalize()} — Confidence: {e['score']:.2f}")
    st.markdown("---")
    st.markdown(friday_reply(top_emotions))

