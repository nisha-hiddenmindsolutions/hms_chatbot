import sys
from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
while PROJECT_ROOT.parent != PROJECT_ROOT and not (PROJECT_ROOT / "data").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

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

    /* App Background (Warm Off-White matching hiddenmindsolutions.in) */
    .stApp {
        background-color: #f4f3ef !important;
        color: #111827 !important;
    }

    .main {
        background-color: #f4f3ef !important;
    }

    .block-container {
        max-width: 960px !important;
        padding-top: 10px !important;
        padding-bottom: 220px !important;
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
        color: #0f172a !important;
    }

    /* Top Navigation Bar */
    .hms-nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 0 24px 0;
        border-bottom: 1px solid #e5e3dc;
        margin-bottom: 28px;
    }

    .hms-logo-container {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 20px;
        font-weight: 600;
        color: #111827;
    }

    .hms-icon {
        width: 32px;
        height: 32px;
        background: #000000;
        color: white;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 14px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
    }

    .hms-badge {
        color: #111827 !important;
        font-weight: 600;
        font-size: 14px;
        background: #ffffff !important;
        padding: 3px 10px;
        border-radius: 6px;
        border: 1px solid #dcdad2;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
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
        background: #0f172a;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 14px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.12);
    }

    /* Hero Greeting Banner */
    .hero-container {
        margin-top: 16px;
        margin-bottom: 40px;
    }

    .greeting-title {
        font-size: 52px;
        font-weight: 600;
        line-height: 1.15;
        letter-spacing: -0.5px;
        margin-bottom: 6px;
        color: #111827;
    }

    .greeting-subtitle {
        font-size: 48px;
        font-weight: 500;
        color: #94a3b8;
        line-height: 1.15;
        letter-spacing: -0.5px;
        margin-bottom: 32px;
    }

    /* Streamlit Chat Messages Styling: Customer on Right, Agent on Left, Zero Avatars/Emojis (Matching Reference) */
    [data-testid^="stChatMessageAvatar"],
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"],
    [data-testid="stChatMessageAvatarCustom"],
    .stChatMessage > div:has([data-testid^="stChatMessageAvatar"]),
    .stChatMessage [data-testid^="stChatMessageAvatar"] {
        display: none !important;
        width: 0 !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    [data-testid="stChatMessage"] {
        background-color: transparent !important;
        padding: 4px 0 !important;
        margin-bottom: 18px !important;
        gap: 0 !important;
        width: 100% !important;
    }

    /* CUSTOMER (USER) MESSAGE: FLUSH RIGHT (NOT IN CENTER) */
    [data-testid="stChatMessage"]:has([aria-label*="user"]),
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        display: flex !important;
        flex-direction: row !important;
        justify-content: flex-end !important;
        padding: 6px 0 !important;
    }

    /* Customer Card: White rounded card aligned to the RIGHT */
    [data-testid="stChatMessageContent"][aria-label*="user"],
    [data-testid="stChatMessage"]:has([aria-label*="user"]) [data-testid="stChatMessageContent"],
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
        background-color: #ffffff !important;
        color: #111827 !important;
        border: 1px solid #e5e3dc !important;
        border-radius: 16px !important;
        padding: 16px 22px !important;
        min-width: 90px !important;
        max-width: 80% !important;
        width: fit-content !important;
        flex-grow: 0 !important;
        flex-shrink: 1 !important;
        margin-left: auto !important;
        margin-right: 0 !important;
        text-align: left !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
        word-break: break-word !important;
    }

    [data-testid="stChatMessageContent"][aria-label*="user"] p,
    [data-testid="stChatMessage"]:has([aria-label*="user"]) [data-testid="stChatMessageContent"] p,
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] p {
        color: #111827 !important;
        margin: 0 !important;
        font-size: 15px !important;
        line-height: 1.6 !important;
        font-weight: 400 !important;
        text-align: left !important;
    }

    /* AGENT (ASSISTANT) MESSAGE: FLUSH LEFT */
    [data-testid="stChatMessage"]:has([aria-label*="assistant"]),
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        display: flex !important;
        flex-direction: row !important;
        justify-content: flex-start !important;
        padding: 8px 0 !important;
    }

    /* Agent Card: Identical format, aligned to Left */
    [data-testid="stChatMessageContent"][aria-label*="assistant"],
    [data-testid="stChatMessage"]:has([aria-label*="assistant"]) [data-testid="stChatMessageContent"],
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {
        background-color: #ffffff !important;
        color: #111827 !important;
        border: 1px solid #e5e3dc !important;
        border-radius: 16px !important;
        padding: 16px 22px 20px 22px !important;
        max-width: 85% !important;
        width: fit-content !important;
        flex-grow: 0 !important;
        flex-shrink: 1 !important;
        margin-left: 0 !important;
        margin-right: auto !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
        word-break: break-word !important;
    }

    [data-testid="stChatMessageContent"][aria-label*="assistant"] p,
    [data-testid="stChatMessage"]:has([aria-label*="assistant"]) [data-testid="stChatMessageContent"] p,
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] p {
        color: #111827 !important;
        margin-bottom: 10px !important;
        font-size: 15px !important;
        line-height: 1.6 !important;
    }

    [data-testid="stChatMessageContent"][aria-label*="assistant"] p:last-child,
    [data-testid="stChatMessage"]:has([aria-label*="assistant"]) [data-testid="stChatMessageContent"] p:last-child,
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] p:last-child,
    [data-testid="stChatMessageContent"][aria-label*="assistant"] ul:last-child,
    [data-testid="stChatMessageContent"][aria-label*="assistant"] ol:last-child {
        margin-bottom: 0 !important;
    }

    [data-testid="stChatMessageContent"][aria-label*="assistant"] ul,
    [data-testid="stChatMessageContent"][aria-label*="assistant"] ol,
    [data-testid="stChatMessage"]:has([aria-label*="assistant"]) [data-testid="stChatMessageContent"] ul,
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] ol {
        color: #111827 !important;
        margin-bottom: 10px !important;
        padding-left: 20px !important;
    }

    [data-testid="stChatMessageContent"][aria-label*="assistant"] li,
    [data-testid="stChatMessage"]:has([aria-label*="assistant"]) [data-testid="stChatMessageContent"] li,
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] li {
        color: #111827 !important;
        margin-bottom: 4px !important;
    }

    /* Markdown container inside chat message: Ensure it wraps content tightly */
    [data-testid="stChatMessageContent"] .stMarkdown,
    [data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] {
        width: fit-content !important;
        max-width: 100% !important;
    }

    /* Floating Pill Input Bar & Entire Bottom Fixed Container (No White Boxes) */
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    .stBottom,
    .stBottom > div,
    [data-testid="stBottomBlockContainer"],
    [data-testid="stChatInputContainer"] {
        background-color: #f4f3ef !important;
        background: #f4f3ef !important;
        border-top: none !important;
    }

    [data-testid="stBottomBlockContainer"] {
        max-width: 960px !important;
        margin: 0 auto !important;
    }

    [data-testid="stChatInputContainer"] {
        max-width: 960px !important;
        margin: 0 auto !important;
    }

    [data-testid="stChatInput"] {
        max-width: 960px !important;
        margin: 0 auto !important;
        background: transparent !important;
    }

    /* Entire Input Pill Background Pure Solid White */
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] > div > div,
    [data-testid="stChatInput"] textarea {
        background-color: #ffffff !important;
        background: #ffffff !important;
    }

    [data-testid="stChatInput"] > div {
        border: 1px solid #dcdad2 !important;
        border-radius: 36px !important;
        padding: 4px 14px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04) !important;
        transition: all 0.2s ease;
    }

    [data-testid="stChatInput"] > div:focus-within {
        border-color: #000000 !important;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.1) !important;
    }

    [data-testid="stChatInput"] textarea {
        color: #111827 !important;
        font-size: 15px !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #64748b !important;
        font-size: 15px !important;
    }

    /* Send Button: Soft slate when idle, turns sleek solid black while typing/focused */
    [data-testid="stChatInput"] button {
        background-color: #c5d3e0 !important;
        color: #ffffff !important;
        border-radius: 50% !important;
        width: 38px !important;
        height: 38px !important;
        border: none !important;
        transition: all 0.25s ease !important;
    }

    [data-testid="stChatInput"] > div:focus-within button {
        background-color: #000000 !important;
        color: #ffffff !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25) !important;
    }

    [data-testid="stChatInput"] button:hover {
        background-color: #000000 !important;
    }

    /* Disclaimer Footer */
    .hms-disclaimer {
        text-align: center;
        font-size: 12px;
        color: #78716c;
        margin-top: 10px;
        padding-bottom: 16px;
    }

    .hms-disclaimer a {
        color: #2563eb;
        text-decoration: underline;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #eeebe5 !important;
        border-right: 1px solid #e0dcd3 !important;
    }

    section[data-testid="stSidebar"] .stButton > button {
        background-color: #ffffff !important;
        border: 1px solid #dcdad2 !important;
        border-radius: 12px !important;
        color: #1e293b !important;
        font-weight: 500 !important;
        padding: 10px 16px !important;
        text-align: left !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02) !important;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: #f4f3ef !important;
        border-color: #0f172a !important;
        color: #0f172a !important;
    }

    /* Card Action Buttons styling */
    .stButton > button {
        border-radius: 12px;
        border: 1px solid #dcdad2;
        background-color: #ffffff;
        color: #111827;
        padding: 12px 18px 14px 18px !important;
        font-weight: 500;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background-color: #0f172a;
        border-color: #0f172a;
        color: #ffffff;
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
# CHAT INPUT PROCESSING & PROMPT CAPTURE
# ============================================================

prompt_input = st.chat_input("Ask about services, hiring, team...")

if st.session_state.active_prompt:
    prompt_input = st.session_state.active_prompt
    st.session_state.active_prompt = None


# ============================================================
# EMPTY STATE: HERO GREETING & HMS PROMPT CARDS GRID
# ============================================================

if not st.session_state.messages and not prompt_input:
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
            <div style="height: 38px; font-weight: 600; font-size: 14px; color: #111827; line-height: 1.35; margin-bottom: 6px; overflow: hidden;">
                Explore Tech & Services
            </div>
            <div style="height: 68px; font-size: 12px; color: #475569; background: #ffffff; padding: 8px 10px; border-radius: 10px; border: 1px solid #dcdad2; font-family: monospace; overflow: hidden; margin-bottom: 12px; line-height: 1.5; box-shadow: 0 1px 3px rgba(0,0,0,0.02);">
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
            <div style="height: 38px; font-weight: 600; font-size: 14px; color: #111827; line-height: 1.35; margin-bottom: 6px; overflow: hidden;">
                Contact & Location
            </div>
            <div style="height: 68px; font-size: 12px; color: #475569; background: #ffffff; padding: 8px 10px; border-radius: 10px; border: 1px solid #dcdad2; font-family: monospace; overflow: hidden; margin-bottom: 12px; line-height: 1.5; box-shadow: 0 1px 3px rgba(0,0,0,0.02);">
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
            <div style="height: 38px; font-weight: 600; font-size: 14px; color: #111827; line-height: 1.35; margin-bottom: 6px; overflow: hidden;">
                Leadership & Core Team
            </div>
            <div style="height: 68px; font-size: 12px; color: #475569; background: #ffffff; padding: 8px 10px; border-radius: 10px; border: 1px solid #dcdad2; font-family: monospace; overflow: hidden; margin-bottom: 12px; line-height: 1.5; box-shadow: 0 1px 3px rgba(0,0,0,0.02);">
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
            <div style="height: 38px; font-weight: 600; font-size: 14px; color: #111827; line-height: 1.35; margin-bottom: 6px; overflow: hidden;">
                Portfolio & Projects
            </div>
            <div style="height: 68px; font-size: 12px; color: #475569; background: #ffffff; padding: 8px 10px; border-radius: 10px; border: 1px solid #dcdad2; font-family: monospace; overflow: hidden; margin-bottom: 12px; line-height: 1.5; box-shadow: 0 1px 3px rgba(0,0,0,0.02);">
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
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ============================================================
# PROCESS NEW PROMPT
# ============================================================

if prompt_input:
    # 1. Display User Message
    with st.chat_message("user"):
        st.markdown(prompt_input)

    # 2. Generate RAG Response
    with st.chat_message("assistant"):
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
