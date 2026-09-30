"""
Gemini Sentiment Analyzer
Built with Python, Streamlit, and the google-genai SDK.
Deployable on Hugging Face Spaces using Docker.
"""

import os
import json
import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


# -----------------------------------------------------------------------------
# 1. Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Gemini Sentiment Analyzer",
    page_icon="🎭",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom modern CSS styling for a clean beginner-friendly interface
st.markdown(
    """
    <style>
    /* Clean font & container styling */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #1E293B;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .sentiment-card {
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        margin-top: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
    }
    .positive-card {
        background-color: #ECFDF5;
        border-left: 6px solid #10B981;
        color: #065F46;
    }
    .negative-card {
        background-color: #FEF2F2;
        border-left: 6px solid #EF4444;
        color: #991B1B;
    }
    .neutral-card {
        background-color: #F1F5F9;
        border-left: 6px solid #64748B;
        color: #334155;
    }
    .badge {
        display: inline-block;
        font-size: 0.85rem;
        font-weight: 600;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    .badge-positive { background-color: #D1FAE5; color: #047857; }
    .badge-negative { background-color: #FEE2E2; color: #B91C1C; }
    .badge-neutral { background-color: #E2E8F0; color: #475569; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 2. Structured Response Schema (Pydantic)
# -----------------------------------------------------------------------------
class SentimentAnalysisResponse(BaseModel):
    sentiment: str = Field(
        description="The classified sentiment: strictly 'Positive', 'Negative', or 'Neutral'"
    )
    confidence: float = Field(
        description="Confidence score of the classification from 0.0 to 1.0"
    )
    explanation: str = Field(
        description="A short, clear 1-2 sentence explanation of why this sentiment was chosen"
    )
    key_phrases: list[str] = Field(
        default_factory=list,
        description="2-4 key words or short phrases that most influenced this sentiment rating"
    )


# -----------------------------------------------------------------------------
# 3. Environment & Gemini Client Setup
# -----------------------------------------------------------------------------
# Requirement: Read the Gemini API key from the GEMINI_API_KEY environment variable.
# Do NOT hardcode the API key.
env_api_key = os.environ.get("GEMINI_API_KEY", "").strip()

# Allow optional manual fallback in the sidebar if running locally without export
with st.sidebar:
    st.header("⚙️ Configuration")
    
    if env_api_key:
        st.success("✅ `GEMINI_API_KEY` detected from environment!")
        active_api_key = env_api_key
    else:
        st.warning("⚠️ `GEMINI_API_KEY` not found in environment.")
        st.info(
            "On **Hugging Face Spaces**, set your key in:\n"
            "**Settings** ➔ **Variables and secrets** ➔ **New secret** (`GEMINI_API_KEY`)."
        )
        active_api_key = st.text_input(
            "Or enter API Key here for testing:",
            type="password",
            help="Your key is kept in memory during this session only.",
        ).strip()

    st.markdown("---")
    st.subheader("📌 Model Details")
    model_name = st.selectbox(
        "Gemini Model",
        options=["gemini-2.5-flash", "gemini-2.0-flash"],
        index=0,
        help="Fast, cost-efficient, and highly accurate for text sentiment classification."
    )
    st.caption("Powered by the modern `google-genai` Python SDK.")
    
    st.markdown("---")
    st.subheader("💡 Deployment Info")
    st.markdown(
        """
        - **Platform:** Hugging Face Spaces (Docker)
        - **Port:** `7860`
        - **Container User:** UID `1000` (`user`)
        """
    )


# -----------------------------------------------------------------------------
# 4. Helper Function: Call Gemini API
# -----------------------------------------------------------------------------
def analyze_sentiment_with_gemini(text: str, api_key: str, model: str) -> SentimentAnalysisResponse:
    """
    Sends the user text to Gemini API using google-genai Python SDK
    and returns a structured SentimentAnalysisResponse.
    """
    client = genai.Client(api_key=api_key)

    prompt = (
        "You are an expert natural language processing assistant. Analyze the sentiment of the text below.\n"
        "Strictly classify the sentiment into exactly one of: 'Positive', 'Negative', or 'Neutral'.\n"
        "Provide a concise, 1-2 sentence explanation highlighting the reasoning, a confidence score between 0.0 and 1.0, "
        "and 2-4 key influential phrases.\n\n"
        f"Text to analyze:\n\"\"\"{text}\"\"\""
    )

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=SentimentAnalysisResponse,
            temperature=0.1,
            system_instruction=(
                "You are an accurate, objective sentiment classification engine. "
                "Classify text as 'Positive', 'Negative', or 'Neutral'. "
                "Always adhere strictly to the JSON schema."
            ),
        ),
    )

    # Parse and validate returned JSON
    raw_json = response.text.strip()
    data = json.loads(raw_json)
    return SentimentAnalysisResponse(**data)


# -----------------------------------------------------------------------------
# 5. User Interface & Workflow
# -----------------------------------------------------------------------------
st.title("🎭 Gemini Sentiment Analyzer")
st.markdown("Enter any sentence, customer review, or paragraph to classify its sentiment with Google Gemini.")

# Preset samples for beginner-friendly exploration
st.markdown("##### ⚡ Quick Example Presets:")
col1, col2, col3, col4 = st.columns(4)

preset_text = ""
if col1.button("🎉 Positive Review", use_container_width=True):
    preset_text = "The customer support team resolved my issue in under ten minutes. Absolutely brilliant service!"
elif col2.button("⚠️ Negative Review", use_container_width=True):
    preset_text = "The package arrived 4 days late, the packaging was torn open, and customer service refused a refund."
elif col3.button("⚖️ Neutral Update", use_container_width=True):
    preset_text = "The quarterly engineering sprint planning meeting has been scheduled for Thursday at 10:00 AM UTC."
elif col4.button("🤔 Mixed Nuance", use_container_width=True):
    preset_text = "The camera hardware and screen resolution are outstanding, but the battery life is surprisingly weak."

# Initialize session state for text input if preset clicked
if preset_text:
    st.session_state["user_input_text"] = preset_text

default_input = st.session_state.get(
    "user_input_text",
    "I was amazed by how smooth and intuitive the onboarding process was. Highly recommended!"
)

# Text input area (Sentence or paragraph)
user_text = st.text_area(
    "Enter a sentence or paragraph to analyze:",
    value=default_input,
    height=120,
    placeholder="Type or paste your text here...",
    help="Can be customer feedback, tweets, emails, news headlines, or paragraphs.",
)

col_btn, col_count = st.columns([1, 2])
with col_btn:
    analyze_clicked = st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True)
with col_count:
    char_len = len(user_text.strip())
    words_len = len(user_text.strip().split()) if char_len > 0 else 0
    st.caption(f"📊 Length: **{words_len}** words | **{char_len}** characters")

# Store analysis history in session state
if "history" not in st.session_state:
    st.session_state["history"] = []

# Processing sentiment analysis
if analyze_clicked:
    if not active_api_key:
        st.error("❌ Please provide a `GEMINI_API_KEY` in your environment or via the sidebar to continue.")
    elif not user_text.strip():
        st.warning("⚠️ Please enter a sentence or paragraph before clicking Analyze Sentiment.")
    else:
        with st.spinner("Analyzing sentiment with Gemini API..."):
            try:
                result = analyze_sentiment_with_gemini(user_text, active_api_key, model_name)
                
                # Standardize sentiment label
                norm_sentiment = result.sentiment.strip().capitalize()
                if norm_sentiment not in ["Positive", "Negative", "Neutral"]:
                    # Fallback normalization
                    if "pos" in norm_sentiment.lower():
                        norm_sentiment = "Positive"
                    elif "neg" in norm_sentiment.lower():
                        norm_sentiment = "Negative"
                    else:
                        norm_sentiment = "Neutral"

                # Emoji & Color theme selection
                theme_map = {
                    "Positive": {"emoji": "😊", "css": "positive-card", "badge": "badge-positive", "color": "#10B981"},
                    "Negative": {"emoji": "😞", "css": "negative-card", "badge": "badge-negative", "color": "#EF4444"},
                    "Neutral":  {"emoji": "😐", "css": "neutral-card",  "badge": "badge-neutral",  "color": "#64748B"},
                }
                theme = theme_map.get(norm_sentiment, theme_map["Neutral"])

                # Display Results Card
                st.markdown("---")
                st.subheader("🎯 Analysis Results")

                st.markdown(
                    f"""
                    <div class="sentiment-card {theme['css']}">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                            <span style="font-size: 1.4rem; font-weight: 700;">
                                {theme['emoji']} {norm_sentiment} Sentiment
                            </span>
                            <span style="font-size: 0.95rem; font-weight: 600;">
                                Confidence: {int(result.confidence * 100)}%
                            </span>
                        </div>
                        <p style="margin-top: 0.5rem; font-size: 1.05rem; line-height: 1.5;">
                            <strong>Explanation:</strong> {result.explanation}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Progress confidence bar
                st.progress(max(0.0, min(1.0, result.confidence)), text=f"Confidence Level: {int(result.confidence * 100)}%")

                # Display Key influential phrases if provided
                if result.key_phrases:
                    st.markdown("##### 🔑 Key Influential Words / Phrases:")
                    badges_html = "".join([f'<span class="badge {theme["badge"]}">{phrase}</span>' for phrase in result.key_phrases])
                    st.markdown(badges_html, unsafe_allow_html=True)

                # Append to session history
                st.session_state["history"].insert(0, {
                    "text": user_text,
                    "sentiment": norm_sentiment,
                    "confidence": result.confidence,
                    "explanation": result.explanation,
                    "emoji": theme["emoji"],
                })

            except Exception as e:
                st.error(f"❌ Gemini API Error: {str(e)}")
                st.info("Tip: Double-check that your `GEMINI_API_KEY` is valid and has access to the Gemini 2.5 Flash model.")

# Display Recent History if available
if st.session_state["history"]:
    with st.expander(f"📜 Session History ({len(st.session_state['history'])} analyses)", expanded=False):
        for idx, item in enumerate(st.session_state["history"][:6]):
            st.markdown(
                f"**{item['emoji']} {item['sentiment']}** ({int(item['confidence']*100)}%)\n"
                f"> *\"{item['text'][:120]}{'...' if len(item['text']) > 120 else ''}\"*\n"
                f"💡 {item['explanation']}"
            )
            if idx < len(st.session_state["history"][:6]) - 1:
                st.divider()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #94A3B8; font-size: 0.85rem;">
        Gemini Sentiment Analyzer • Python & Streamlit • Powered by Google GenAI SDK (`google-genai`) • Hugging Face Spaces Ready
    </div>
    """,
    unsafe_allow_html=True,
)
