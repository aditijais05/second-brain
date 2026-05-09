"""
Second Brain — Streamlit UI

Run:
    streamlit run app.py

Pages:
    Chat      — ask questions, get cited answers
    Timeline  — see your knowledge base sorted by date
    Ingest    — add new documents
"""

import streamlit as st

st.set_page_config(
    page_title="Second Brain",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d0d0d;
    border-right: 1px solid #1e1e1e;
}
section[data-testid="stSidebar"] * {
    color: #e8e4dc !important;
}
section[data-testid="stSidebar"] .stRadio label {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #888 !important;
}
section[data-testid="stSidebar"] .stRadio div[data-baseweb="radio"] {
    gap: 0.2rem;
}

/* Main area */
.main .block-container {
    padding-top: 2rem;
    max-width: 860px;
}

/* Brand header */
.brand {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: #0d0d0d;
    letter-spacing: -0.02em;
    margin-bottom: 0;
    line-height: 1;
}
.brand-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #888;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 0.2rem;
    margin-bottom: 2rem;
}

/* Chat messages */
.msg-user {
    background: #0d0d0d;
    color: #e8e4dc;
    border-radius: 16px 16px 4px 16px;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 0;
    margin-left: 15%;
    font-size: 0.95rem;
    line-height: 1.6;
}
.msg-assistant {
    background: #f5f2ec;
    color: #1a1a1a;
    border-radius: 4px 16px 16px 16px;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 0;
    margin-right: 15%;
    font-size: 0.95rem;
    line-height: 1.7;
    border-left: 3px solid #c8b89a;
}

/* Citation badges */
.citation {
    display: inline-block;
    background: #0d0d0d;
    color: #e8e4dc;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    padding: 1px 6px;
    border-radius: 4px;
    margin: 0 1px;
    vertical-align: super;
    cursor: pointer;
    text-decoration: none;
}

/* Source cards */
.source-card {
    background: #faf8f4;
    border: 1px solid #e8e2d8;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.85rem;
}
.source-num {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.source-title {
    font-weight: 500;
    color: #1a1a1a;
    margin: 0.2rem 0;
}
.source-preview {
    color: #666;
    font-size: 0.82rem;
    font-style: italic;
    line-height: 1.5;
}

/* Timeline */
.timeline-item {
    display: flex;
    gap: 1.2rem;
    margin-bottom: 1.5rem;
    align-items: flex-start;
}
.timeline-date {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #888;
    min-width: 90px;
    padding-top: 0.2rem;
    text-align: right;
    letter-spacing: 0.04em;
}
.timeline-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #0d0d0d;
    margin-top: 0.35rem;
    flex-shrink: 0;
}
.timeline-line {
    border-left: 1px solid #e0dbd2;
    margin-left: 4px;
    padding-left: 1.2rem;
    flex: 1;
}
.timeline-card {
    background: #faf8f4;
    border: 1px solid #e8e2d8;
    border-radius: 8px;
    padding: 0.8rem 1rem;
}
.timeline-card-title {
    font-weight: 500;
    font-size: 0.95rem;
    color: #1a1a1a;
}
.timeline-card-meta {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #aaa;
    margin-top: 0.2rem;
    letter-spacing: 0.04em;
}
.timeline-card-preview {
    font-size: 0.83rem;
    color: #666;
    margin-top: 0.4rem;
    line-height: 1.5;
}

/* Type badges */
.badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    padding: 2px 7px;
    border-radius: 4px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-left: 0.4rem;
}
.badge-pdf      { background: #fde8e0; color: #c05a3a; }
.badge-markdown { background: #e0ede8; color: #2d7a5a; }
.badge-url      { background: #e0e8f5; color: #2d4d9a; }
.badge-notion   { background: #ede8f5; color: #5a3d9a; }

/* Input area */
.stTextInput input {
    border-radius: 12px !important;
    border: 1.5px solid #e0dbd2 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.7rem 1rem !important;
}
.stTextInput input:focus {
    border-color: #0d0d0d !important;
    box-shadow: none !important;
}

/* Buttons */
.stButton button {
    background: #0d0d0d !important;
    color: #e8e4dc !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    padding: 0.5rem 1.2rem !important;
}
.stButton button:hover {
    background: #2a2a2a !important;
}

/* Stat cards */
.stat-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 2rem;
}
.stat-card {
    flex: 1;
    background: #faf8f4;
    border: 1px solid #e8e2d8;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.stat-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: #0d0d0d;
    line-height: 1;
}
.stat-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.3rem;
}

/* Expander */
.streamlit-expanderHeader {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    color: #888 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

/* Divider */
hr { border-color: #e8e2d8; margin: 1.5rem 0; }

/* Hide default Streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "pipeline":    None,
        "docs":        [],
        "chat_history": [],
        "indexed":     False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🧠 Second Brain")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["💬  Chat", "📅  Timeline", "➕  Ingest"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Pipeline status
    if st.session_state.indexed:
        doc_count   = len(st.session_state.docs)
        chunk_count = st.session_state.pipeline.vector_store.count()
        st.markdown(f"""
        <div style='font-family: DM Mono, monospace; font-size: 0.7rem; color: #4a4; letter-spacing: 0.06em;'>
        ● INDEXED<br>
        {doc_count} docs · {chunk_count} chunks
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='font-family: DM Mono, monospace; font-size: 0.7rem; color: #888; letter-spacing: 0.06em;'>
        ○ NOT INDEXED<br>
        Add docs in Ingest tab
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Settings
    with st.expander("Settings"):
        embedding_backend = st.selectbox(
            "Embedding model",
            ["local", "openai"],
            help="local = free, openai = better quality"
        )
        top_k = st.slider("Results per query", 3, 10, 5)
        import os
        saved_key = st.session_state.get("gemini_key", os.environ.get("GEMINI_API_KEY", ""))
        gemini_key = st.text_input(
            "Gemini API key",
            value=saved_key,
            type="password",
            placeholder="AIza...",
            help="Get free key at aistudio.google.com"
        )
        if gemini_key:
            os.environ["GEMINI_API_KEY"] = gemini_key
            st.session_state["gemini_key"] = gemini_key


# ── Page routing ───────────────────────────────────────────────────────────

if "Chat" in page:
    from ui.chat_page import render_chat
    render_chat(top_k=top_k)

elif "Timeline" in page:
    from ui.timeline_page import render_timeline
    render_timeline()

elif "Ingest" in page:
    from ui.ingest_page import render_ingest
    render_ingest(embedding_backend=embedding_backend)