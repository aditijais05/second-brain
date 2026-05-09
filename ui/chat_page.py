"""Chat page — ask questions, get cited answers."""

import re
import os
import streamlit as st


def render_chat(top_k: int = 5):

    st.markdown('<div class="brand">Ask anything</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">Cited answers from your knowledge base</div>', unsafe_allow_html=True)

    # ── Guard: not indexed ─────────────────────────────────────────────
    if not st.session_state.indexed:
        st.info("Your knowledge base is empty. Go to **Ingest** to add documents first.")
        return

    # ── Guard: no API key ──────────────────────────────────────────────
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not gemini_key:
        st.warning("Enter your Gemini API key in the sidebar Settings before asking questions.")
        return

    # ── Chat history ───────────────────────────────────────────────────
    for entry in st.session_state.chat_history:
        st.markdown(f'<div class="msg-user">{entry["question"]}</div>', unsafe_allow_html=True)
        _render_answer(entry["answer"])
        st.markdown("<hr>", unsafe_allow_html=True)

    # ── Input ──────────────────────────────────────────────────────────
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            question = st.text_input(
                "Question",
                placeholder="What does your knowledge base say about...",
                label_visibility="collapsed",
            )
        with col2:
            submitted = st.form_submit_button("Ask ->")

    if submitted and question.strip():
        with st.spinner("Thinking..."):
            try:
                # Always reinitialise the generator with the current key
                # so sidebar key changes are picked up immediately
                from generation.generator import CitationGenerator
                st.session_state.pipeline.generator = CitationGenerator(
                    api_key=gemini_key
                )
                answer = st.session_state.pipeline.ask(
                    question.strip(), top_k=top_k
                )
                st.session_state.chat_history.append({
                    "question": question.strip(),
                    "answer":   answer,
                })
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    # ── Clear button ───────────────────────────────────────────────────
    if st.session_state.chat_history:
        if st.button("Clear chat"):
            st.session_state.chat_history = []
            st.rerun()


def _render_answer(answer):
    """Render answer text with inline citation badges + expandable source cards."""
    text = answer.answer
    text = re.sub(
        r"\[source_(\d+)\]",
        r'<span class="citation">[\1]</span>',
        text
    )
    st.markdown(f'<div class="msg-assistant">{text}</div>', unsafe_allow_html=True)

    if answer.cited_sources:
        with st.expander(f"Sources ({len(answer.cited_sources)})"):
            for s in answer.cited_sources:
                badge = _source_type_badge(s.source_uri)
                st.markdown(f"""
                <div class="source-card">
                    <div class="source-num">[{s.source_num}] {badge}</div>
                    <div class="source-title">{s.title}</div>
                    <div class="source-preview">"{s.text[:200].strip()}..."</div>
                    <div style="font-family: DM Mono, monospace; font-size: 0.65rem;
                                color: #bbb; margin-top: 0.4rem;">{s.source_uri}</div>
                </div>
                """, unsafe_allow_html=True)


def _source_type_badge(uri: str) -> str:
    uri = uri.lower()
    if uri.endswith(".pdf"):
        return '<span class="badge badge-pdf">PDF</span>'
    if uri.endswith(".md") or uri.endswith(".txt"):
        return '<span class="badge badge-markdown">MD</span>'
    if uri.startswith("http"):
        return '<span class="badge badge-url">URL</span>'
    return '<span class="badge badge-markdown">DOC</span>'