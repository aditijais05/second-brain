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

st.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Syne:wght@700;800&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: #0a0a0f;
    color: #f0eeff;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0d0d15 !important;
    border-right: 1px solid #1e1a2e;
}
section[data-testid="stSidebar"] * {
    color: #c4b8f0 !important;
}
section[data-testid="stSidebar"] .stRadio label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #7b6fa0 !important;
}

/* ── Main area ── */
.main {
    background-color: #0a0a0f;
}
.main .block-container {
    padding-top: 2.5rem;
    max-width: 880px;
    background-color: #0a0a0f;
}

/* ── Brand header ── */
.brand {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.03em;
    margin-bottom: 0;
    line-height: 1;
    background: linear-gradient(135deg, #ffffff 0%, #b388ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.brand-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #6b5fa0;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-top: 0.3rem;
    margin-bottom: 2.5rem;
}

/* ── Chat messages ── */
.msg-user {
    background: linear-gradient(135deg, #6c3bcc 0%, #4a1fa8 100%);
    color: #f0eeff;
    border-radius: 18px 18px 4px 18px;
    padding: 0.9rem 1.3rem;
    margin: 0.6rem 0;
    margin-left: 12%;
    font-size: 0.95rem;
    line-height: 1.6;
    box-shadow: 0 4px 24px rgba(108, 59, 204, 0.3);
}
.msg-assistant {
    background: #13111f;
    color: #e8e0ff;
    border-radius: 4px 18px 18px 18px;
    padding: 0.9rem 1.3rem;
    margin: 0.6rem 0;
    margin-right: 12%;
    font-size: 0.95rem;
    line-height: 1.7;
    border-left: 3px solid #7c4dff;
    box-shadow: 0 2px 16px rgba(0,0,0,0.4);
}

/* ── Citation badges ── */
.citation {
    display: inline-block;
    background: #7c4dff;
    color: #ffffff;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    padding: 1px 6px;
    border-radius: 4px;
    margin: 0 2px;
    vertical-align: super;
    cursor: pointer;
    font-weight: 500;
    letter-spacing: 0.04em;
}

/* ── Source cards ── */
.source-card {
    background: #0f0d1a;
    border: 1px solid #1e1a2e;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin: 0.5rem 0;
    font-size: 0.85rem;
    transition: border-color 0.2s;
}
.source-card:hover {
    border-color: #7c4dff;
}
.source-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #6b5fa0;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.source-title {
    font-weight: 600;
    color: #f0eeff;
    margin: 0.25rem 0;
    font-size: 0.9rem;
}
.source-preview {
    color: #8b7db0;
    font-size: 0.82rem;
    font-style: italic;
    line-height: 1.5;
}

/* ── Timeline ── */
.timeline-item {
    display: flex;
    gap: 1.2rem;
    margin-bottom: 1.5rem;
    align-items: flex-start;
}
.timeline-date {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #6b5fa0;
    min-width: 90px;
    padding-top: 0.25rem;
    text-align: right;
    letter-spacing: 0.04em;
}
.timeline-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #7c4dff;
    margin-top: 0.35rem;
    flex-shrink: 0;
    box-shadow: 0 0 8px rgba(124, 77, 255, 0.6);
}
.timeline-line {
    border-left: 1px solid #1e1a2e;
    margin-left: 4px;
    padding-left: 1.2rem;
    flex: 1;
}
.timeline-card {
    background: #0f0d1a;
    border: 1px solid #1e1a2e;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
}
.timeline-card-title {
    font-weight: 600;
    font-size: 0.95rem;
    color: #f0eeff;
}
.timeline-card-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.67rem;
    color: #6b5fa0;
    margin-top: 0.2rem;
    letter-spacing: 0.04em;
}
.timeline-card-preview {
    font-size: 0.83rem;
    color: #8b7db0;
    margin-top: 0.4rem;
    line-height: 1.5;
}

/* ── Type badges ── */
.badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    padding: 2px 7px;
    border-radius: 4px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-left: 0.4rem;
    font-weight: 500;
}
.badge-pdf      { background: #2a1020; color: #f06292; border: 1px solid #4a1530; }
.badge-markdown { background: #0d2218; color: #69f0ae; border: 1px solid #1a4030; }
.badge-url      { background: #0d1a2e; color: #40c4ff; border: 1px solid #1a2e50; }
.badge-notion   { background: #1a1228; color: #b388ff; border: 1px solid #2e1e50; }

/* ── Input area ── */
.stTextInput input {
    border-radius: 12px !important;
    border: 1.5px solid #1e1a2e !important;
    background: #0f0d1a !important;
    color: #f0eeff !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.75rem 1rem !important;
}
.stTextInput input:focus {
    border-color: #7c4dff !important;
    box-shadow: 0 0 0 3px rgba(124, 77, 255, 0.15) !important;
}
.stTextInput input::placeholder {
    color: #4a4060 !important;
}

/* ── Buttons ── */
.stButton button {
    background: #7c4dff !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 0.55rem 1.3rem !important;
    font-weight: 500 !important;
    transition: background 0.2s, box-shadow 0.2s !important;
}
.stButton button:hover {
    background: #9c6fff !important;
    box-shadow: 0 4px 20px rgba(124, 77, 255, 0.4) !important;
}

/* ── Stat cards ── */
.stat-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 2rem;
}
.stat-card {
    flex: 1;
    background: #0f0d1a;
    border: 1px solid #1e1a2e;
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    text-align: center;
}
.stat-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: #b388ff;
    line-height: 1;
}
.stat-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.63rem;
    color: #6b5fa0;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: 0.35rem;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.73rem !important;
    color: #6b5fa0 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}

/* ── Divider ── */
hr {
    border-color: #1e1a2e;
    margin: 1.5rem 0;
}

/* ── Info / warning boxes ── */
.stAlert {
    background: #0f0d1a !important;
    border: 1px solid #1e1a2e !important;
    border-radius: 10px !important;
    color: #c4b8f0 !important;
}

/* ── Selectbox / slider ── */
.stSelectbox > div > div {
    background: #0f0d1a !important;
    border-color: #1e1a2e !important;
    color: #f0eeff !important;
    border-radius: 10px !important;
}
.stSlider .st-bx {
    background: #7c4dff !important;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: #2a2040; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #7c4dff; }
</style>
""")


# ── Session state ──────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "pipeline":     None,
        "docs":         [],
        "chat_history": [],
        "indexed":      False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.html("""
    <div style='padding: 0.5rem 0 1rem;'>
        <div style='font-family: Syne, sans-serif; font-size: 1.4rem; font-weight: 800;
                    background: linear-gradient(135deg, #ffffff, #b388ff);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    background-clip: text; letter-spacing: -0.02em;'>
            🧠 Second Brain
        </div>
        <div style='font-family: JetBrains Mono, monospace; font-size: 0.6rem;
                    color: #4a4060; letter-spacing: 0.12em; text-transform: uppercase;
                    margin-top: 0.2rem;'>
            Personal Knowledge OS
        </div>
    </div>
    """)

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
        st.html(f"""
        <div style='font-family: JetBrains Mono, monospace; font-size: 0.68rem;
                    color: #69f0ae; letter-spacing: 0.08em; line-height: 1.8;'>
            ● INDEXED<br>
            <span style='color: #4a9060;'>{doc_count} docs · {chunk_count} chunks</span>
        </div>
        """)
    else:
        st.html("""
        <div style='font-family: JetBrains Mono, monospace; font-size: 0.68rem;
                    color: #4a4060; letter-spacing: 0.08em; line-height: 1.8;'>
            ○ NOT INDEXED<br>
            Add docs in Ingest tab
        </div>
        """)

    st.markdown("---")

    with st.expander("⚙ Settings"):
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