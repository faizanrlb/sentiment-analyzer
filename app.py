import streamlit as st
from transformers import pipeline

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="💬",
    layout="centered",
)

# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
        .block-container {
            max-width: 800px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .main-title {
            font-size: 2.4rem;
            font-weight: 800;
            margin-bottom: 0.3rem;
        }

        .subtitle {
            color: #6B7280;
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
        }

        .result-box {
            padding: 1.2rem;
            border-radius: 14px;
            border: 1px solid rgba(128, 128, 128, 0.25);
            margin-top: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Load the Hugging Face model once and cache it
# ---------------------------------------------------------
MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
    )

try:
    sentiment_model = load_sentiment_model()
except Exception as error:
    st.error("The sentiment model could not be loaded.")
    st.exception(error)
    st.stop()

# ---------------------------------------------------------
# App UI
# ---------------------------------------------------------
st.markdown('<div class="main-title">💬 Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Enter English text and the AI model will classify it as positive or negative.</div>',
    unsafe_allow_html=True,
)

example_text = "I really enjoyed using this application. It was simple and helpful."

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Use example"):
        st.session_state.input_text = example_text

with col2:
    if st.button("Clear"):
        st.session_state.input_text = ""

text = st.text_area(
    "Text to analyze",
    key="input_text",
    height=180,
    placeholder="Example: The service was excellent and I am very satisfied.",
)

st.caption(f"{len(text)} characters")

analyze = st.button("Analyze Sentiment", type="primary", use_container_width=True)

if analyze:
    cleaned_text = text.strip()

    if not cleaned_text:
        st.warning("Please enter some text before analyzing.")
    else:
        with st.spinner("Analyzing sentiment..."):
            try:
                result = sentiment_model(cleaned_text, truncation=True)[0]
                label = result["label"].upper()
                confidence = float(result["score"])

                if label == "POSITIVE":
                    emoji = "😊"
                    friendly_label = "Positive"
                else:
                    emoji = "😞"
                    friendly_label = "Negative"

                st.markdown(
                    f"""
                    <div class="result-box">
                        <h3>{emoji} {friendly_label}</h3>
                        <p><strong>Confidence:</strong> {confidence:.2%}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.progress(confidence)

                with st.expander("Technical details"):
                    st.write(f"Model: `{MODEL_NAME}`")
                    st.json(result)

            except Exception as error:
                st.error("Something went wrong while analyzing the text.")
                st.exception(error)

st.divider()
st.caption(
    "Model: DistilBERT fine-tuned on SST-2. "
    "This demo supports English positive/negative sentiment classification."
)
