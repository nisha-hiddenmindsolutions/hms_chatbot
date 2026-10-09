import sys
from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

import os
import streamlit as st

# Sync Streamlit Cloud secrets to os.environ
try:
    if "GEMINI_API_KEY" in st.secrets and not os.getenv("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

from app.ingestion.document_loader import load_documents
from app.ingestion.document_cleaner import clean_documents
from app.ingestion.document_chunker import chunk_documents
from app.rag.rag_pipeline import (
    create_rag_pipeline,
    generate_answer,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Hidden Mind Solutions AI Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# REAL CHATBOT STYLING (HIDDEN MIND SOLUTIONS AI ASSISTANT)
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* App Background */
    .stApp {
        background-color: #ffffff !important;
        color: #1f1f1f !important;
    }

    .main {
        background-color: #ffffff !important;
    }

    .block-container {
        max-width: 960px !important;
        padding-top: 10px !important;
        padding-bottom: 120px !important;
    }

    /* Hide standard footer & main menu, but keep sidebar collapse toggle visible */
    #MainMenu, footer {
        visibility: hidden !important;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    [data-testid="collapsedControl"], [data-testid="stSidebarCollapseButton"] {
        visibility: visible !important;
        display: flex !important;
        color: #2563eb !important;
    }

    /* Top Navigation Bar */
    .hms-nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 0 24px 0;
        border-bottom: 1px solid #f0f4f9;
        margin-bottom: 28px;
    }

    .hms-logo-container {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 20px;
        font-weight: 600;
        color: #1f1f1f;
    }

    .hms-icon {
        width: 32px;
        height: 32px;
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: white;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 14px;
        box-shadow: 0 2px 6px rgba(37, 99, 235, 0.3);
    }

    .hms-badge {
        color: #2563eb;
        font-weight: 600;
        font-size: 15px;
        background: #eff6ff;
        padding: 3px 8px;
        border-radius: 6px;
        border: 1px solid #bfdbfe;
    }

    .hms-user-profile {
        display: flex;
        align-items: center;
        gap: 16px;
    }

    .user-avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background: linear-gradient(135deg, #1e293b, #0f172a);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 14px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    }

    /* Hero Greeting Banner */
    .hero-container {
        margin-top: 16px;
        margin-bottom: 40px;
    }

    .greeting-title {
        font-size: 52px;
        font-weight: 500;
        line-height: 1.15;
        letter-spacing: -0.5px;
        margin-bottom: 6px;
        background: linear-gradient(90deg, #2563eb 0%, #3b82f6 40%, #1d4ed8 80%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .greeting-subtitle {
        font-size: 48px;
        font-weight: 500;
        color: #c4c7c5;
        line-height: 1.15;
        letter-spacing: -0.5px;
        margin-bottom: 32px;
    }

    /* Streamlit Chat Messages Styling */
    [data-testid="stChatMessage"] {
        background-color: transparent !important;
        padding: 12px 0 !important;
    }

    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) > div:nth-child(2) {
        background-color: #f0f4f9 !important;
        color: #1f1f1f !important;
        border-radius: 20px 20px 4px 20px !important;
        padding: 14px 18px !important;
        max-width: 80% !important;
        margin-left: auto !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) > div:nth-child(2) {
        background-color: #ffffff !important;
        color: #1f1f1f !important;
        border: 1px solid #e3e3e3 !important;
        border-radius: 20px 20px 20px 4px !important;
        padding: 16px 20px !important;
        max-width: 85% !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    /* Floating Pill Input Bar - Aligned Exactly from Explore Services (Card 1) to View Portfolio (Card 4) */
    [data-testid="stBottom"] {
        background-color: #ffffff !important;
    }

    [data-testid="stBottomBlockContainer"] {
        background-color: #ffffff !important;
        border-top: none !important;
        max-width: 960px !important;
        margin: 0 auto !important;
    }

    [data-testid="stChatInputContainer"] {
        background-color: #ffffff !important;
        max-width: 960px !important;
        margin: 0 auto !important;
    }

    [data-testid="stChatInput"] {
        max-width: 960px !important;
        margin: 0 auto !important;
    }

    [data-testid="stChatInput"] > div {
        background-color: #f0f4f9 !important;
        border: 1px solid #e3e3e3 !important;
        border-radius: 28px !important;
        padding: 4px 12px !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06) !important;
        transition: all 0.2s ease;
    }

    [data-testid="stChatInput"] > div:focus-within {
        border-color: #93c5fd !important;
        background-color: #ffffff !important;
        box-shadow: 0 4px 20px rgba(37, 99, 235, 0.15) !important;
    }

    [data-testid="stChatInput"] textarea {
        color: #1f1f1f !important;
        font-size: 16px !important;
        background: transparent !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #5f6368 !important;
    }

    [data-testid="stChatInput"] button {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border-radius: 50% !important;
        width: 38px !important;
        height: 38px !important;
    }

    /* Disclaimer Footer */
    .hms-disclaimer {
        text-align: center;
        font-size: 12px;
        color: #747775;
        margin-top: 10px;
    }

    .hms-disclaimer a {
        color: #2563eb;
        text-decoration: underline;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #f8fafc !important;
        border-right: 1px solid #e2e8f0 !important;
    }

    section[data-testid="stSidebar"] .stButton > button {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 14px;
        color: #1e293b;
        font-weight: 500;
        padding: 10px 16px;
        text-align: left;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: #eff6ff;
        border-color: #3b82f6;
        color: #1d4ed8;
    }

    /* Card Action Buttons styling override */
    .stButton > button {
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        background-color: #f8fafc;
        color: #1e293b;
        padding: 12px;
        font-weight: 500;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background-color: #eff6ff;
        border-color: #2563eb;
        color: #1d4ed8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "active_prompt" not in st.session_state:
    st.session_state.active_prompt = None


# ============================================================
# MEMORY ADAPTER
# ============================================================

class StreamlitMemory:
    def __init__(self):
        self.messages = []

    def get_history(self):
        return self.messages

    def add_user_message(self, content):
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content):
        self.messages.append({"role": "assistant", "content": content})

    def clear(self):
        self.messages = []


if "memory" not in st.session_state:
    st.session_state.memory = StreamlitMemory()


def reset_chat():
    st.session_state.messages = []
    st.session_state.memory.clear()


def set_prompt(prompt_text):
    st.session_state.active_prompt = prompt_text


# ============================================================
# TOP NAV BAR
# ============================================================

st.markdown(
    """
    <div class="hms-nav">
        <div class="hms-logo-container">
            <div class="hms-icon">HMS</div>
            <span>Hidden Mind Solutions</span>
            <span class="hms-badge">AI Assistant</span>
        </div>
        <div class="hms-user-profile">
            <div class="user-avatar">HMS</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# INITIALIZE RAG PIPELINE
# ============================================================

@st.cache_resource
def initialize_rag():
    documents = load_documents()
    cleaned_documents = clean_documents(documents)
    chunks = chunk_documents(cleaned_documents)
    client, retriever = create_rag_pipeline(chunks)
    return client, retriever


try:
    client, retriever = initialize_rag()
except Exception as e:
    st.error("Failed to initialize the Hidden Mind Solutions AI Assistant.")
    st.exception(e)
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("### 🧠 HMS AI Assistant")
    st.caption("Grounded strictly on Hidden Mind Solutions Knowledge Base.")

    if st.button("➕  New chat", use_container_width=True):
        reset_chat()
        st.rerun()

    st.divider()
    st.markdown("##### 📌 Sample Questions")

    sample_questions = [
        "What services does Hidden Mind Solutions provide?",
        "Who is the founder and CEO of Hidden Mind Solutions?",
        "What technologies are in the company stack?",
        "Where is the headquarters office located?",
        "What are the payment terms and GST rules?",
    ]

    for q in sample_questions:
        st.button(
            q,
            key=f"sidebar_q_{q}",
            on_click=set_prompt,
            args=(q,),
            use_container_width=True,
        )


# ============================================================
# EMPTY STATE: HERO GREETING & HMS PROMPT CARDS GRID
# ============================================================

if not st.session_state.messages:
    st.markdown(
        """
        <div class="hero-container">
            <div class="greeting-title">Hello!</div>
            <div class="greeting-subtitle">How can I help you today?</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Render 4 Prompt Suggestion Cards for HMS Website (100% Uniform & Baseline Aligned)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
            <div style="height: 38px; font-weight: 600; font-size: 14px; color: #1e293b; line-height: 1.35; margin-bottom: 6px; overflow: hidden;">
                Explore Tech & Services
            </div>
            <div style="height: 68px; font-size: 12px; color: #475569; background: #f8fafc; padding: 8px 10px; border-radius: 10px; border: 1px solid #e2e8f0; font-family: monospace; overflow: hidden; margin-bottom: 12px; line-height: 1.5;">
                1. Web & AI Services<br>
                2. Python & MERN Stack<br>
                3. Cloud & DevOps
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Explore Services", key="card1", use_container_width=True):
            set_prompt("What services does Hidden Mind Solutions provide?")
            st.rerun()

    with col2:
        st.markdown(
            """
            <div style="height: 38px; font-weight: 600; font-size: 14px; color: #1e293b; line-height: 1.35; margin-bottom: 6px; overflow: hidden;">
                Contact & Location
            </div>
            <div style="height: 68px; font-size: 12px; color: #475569; background: #f8fafc; padding: 8px 10px; border-radius: 10px; border: 1px solid #e2e8f0; font-family: monospace; overflow: hidden; margin-bottom: 12px; line-height: 1.5;">
                📍 Udaipur Headquarters<br>
                ✉️ info@hiddenmindsolutions.in<br>
                📞 +91 9376778747
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Contact Details", key="card2", use_container_width=True):
            set_prompt("Where is Hidden Mind Solutions located and how can I contact them?")
            st.rerun()

    with col3:
        st.markdown(
            """
            <div style="height: 38px; font-weight: 600; font-size: 14px; color: #1e293b; line-height: 1.35; margin-bottom: 6px; overflow: hidden;">
                Leadership & Core Team
            </div>
            <div style="height: 68px; font-size: 12px; color: #475569; background: #f8fafc; padding: 8px 10px; border-radius: 10px; border: 1px solid #e2e8f0; font-family: monospace; overflow: hidden; margin-bottom: 12px; line-height: 1.5;">
                • Himanshu Sanadhya (CEO)<br>
                • Tushar Vaghela (CTO)<br>
                • Core Team & Leadership
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Meet Leadership", key="card3", use_container_width=True):
            set_prompt("Who is the founder and leadership team of Hidden Mind Solutions?")
            st.rerun()

    with col4:
        st.markdown(
            """
            <div style="height: 38px; font-weight: 600; font-size: 14px; color: #1e293b; line-height: 1.35; margin-bottom: 6px; overflow: hidden;">
                Portfolio & Projects
            </div>
            <div style="height: 68px; font-size: 12px; color: #475569; background: #f8fafc; padding: 8px 10px; border-radius: 10px; border: 1px solid #e2e8f0; font-family: monospace; overflow: hidden; margin-bottom: 12px; line-height: 1.5;">
                • 25+ Delivered Projects<br>
                • Real Estate & SaaS Apps<br>
                • AI & Web Applications
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("View Portfolio", key="card4", use_container_width=True):
            set_prompt("What featured projects has Hidden Mind Solutions delivered?")
            st.rerun()

    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)


# ============================================================
# DISPLAY EXISTING CHAT MESSAGES
# ============================================================

for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🧠"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])


# ============================================================
# CHAT INPUT PROCESSING
# ============================================================

prompt_input = st.chat_input("Enter a prompt here")

if st.session_state.active_prompt:
    prompt_input = st.session_state.active_prompt
    st.session_state.active_prompt = None


if prompt_input:
    # 1. Display User Message
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt_input)

    # 2. Generate RAG Response
    with st.chat_message("assistant", avatar="🧠"):
        with st.spinner("HMS AI Assistant is thinking..."):
            try:
                response = generate_answer(
                    question=prompt_input,
                    client=client,
                    retriever=retriever,
                    memory=st.session_state.memory,
                )
            except Exception as ex:
                response = "I'm sorry, an error occurred while generating the answer."

        st.markdown(response)

    # 3. Save to Session Messages
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    st.session_state.messages.append({"role": "assistant", "content": response})


# ============================================================
# FOOTER DISCLAIMER
# ============================================================

st.markdown(
    """
    <div class="hms-disclaimer">
        Hidden Mind Solutions AI Assistant may display info that should be verified. 
        Learn more at <a href="https://hiddenmindsolutions.in/" target="_blank">hiddenmindsolutions.in</a>
    </div>
    """,
    unsafe_allow_html=True,
)
