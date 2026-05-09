"""Timeline page — knowledge base sorted chronologically."""

import streamlit as st
from collections import defaultdict


def render_timeline():

    st.markdown('<div class="brand">Knowledge timeline</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">Your documents, sorted by date</div>', unsafe_allow_html=True)

    # ── Guard ──────────────────────────────────────────────────────────
    if not st.session_state.indexed or not st.session_state.docs:
        st.info("No documents indexed yet. Go to **Ingest** to add some.")
        return

    docs = st.session_state.docs

    # ── Stats row ──────────────────────────────────────────────────────
    chunk_count = st.session_state.pipeline.vector_store.count()
    types       = [d.source_type.value for d in docs]

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card">
            <div class="stat-value">{len(docs)}</div>
            <div class="stat-label">Documents</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{chunk_count}</div>
            <div class="stat-label">Chunks indexed</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{types.count("pdf")}</div>
            <div class="stat-label">PDFs</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{types.count("url")}</div>
            <div class="stat-label">Web pages</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{types.count("markdown")}</div>
            <div class="stat-label">Notes</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Filter bar ─────────────────────────────────────────────────────
    col1, col2 = st.columns([3, 1])
    with col1:
        search = st.text_input(
            "Filter", placeholder="Search by title or tag…",
            label_visibility="collapsed"
        )
    with col2:
        type_filter = st.selectbox(
            "Type", ["All", "PDF", "Markdown", "URL"],
            label_visibility="collapsed"
        )

    # ── Sort and filter ────────────────────────────────────────────────
    filtered = docs
    if search:
        q = search.lower()
        filtered = [
            d for d in filtered
            if q in d.title.lower()
            or q in d.content[:500].lower()
            or any(q in t for t in d.tags)
        ]
    if type_filter != "All":
        type_map = {"PDF": "pdf", "Markdown": "markdown", "URL": "url"}
        filtered = [d for d in filtered if d.source_type.value == type_map[type_filter]]

    sorted_docs = sorted(filtered, key=lambda d: d.created_at, reverse=True)

    if not sorted_docs:
        st.warning("No documents match your filter.")
        return

    st.markdown(f"<hr>", unsafe_allow_html=True)

    # ── Group by month ─────────────────────────────────────────────────
    by_month = defaultdict(list)
    for doc in sorted_docs:
        month_key = doc.created_at.strftime("%B %Y")
        by_month[month_key].append(doc)

    for month, month_docs in by_month.items():
        st.markdown(f"""
        <div style="font-family: DM Mono, monospace; font-size: 0.7rem;
                    color: #888; text-transform: uppercase; letter-spacing: 0.1em;
                    margin: 1.5rem 0 0.8rem;">
            {month}
        </div>
        """, unsafe_allow_html=True)

        for doc in month_docs:
            badge    = _type_badge(doc.source_type.value)
            date_str = doc.created_at.strftime("%d %b")
            preview  = doc.content[:180].replace("\n", " ").strip()
            tags_html = "".join(
                f'<span style="background:#f0ede6; color:#666; font-size:0.65rem; '
                f'padding:2px 6px; border-radius:4px; margin-right:4px; '
                f'font-family: DM Mono, monospace;">{t}</span>'
                for t in doc.tags[:4]
            )

            st.markdown(f"""
            <div class="timeline-item">
                <div class="timeline-date">{date_str}</div>
                <div>
                    <div class="timeline-dot"></div>
                </div>
                <div class="timeline-line">
                    <div class="timeline-card">
                        <div class="timeline-card-title">
                            {doc.title} {badge}
                        </div>
                        <div class="timeline-card-meta">
                            {doc.source_uri} &nbsp;·&nbsp; {doc.word_count} words
                            {f" &nbsp;·&nbsp; {doc.author}" if doc.author else ""}
                        </div>
                        {f'<div style="margin-top:0.3rem;">{tags_html}</div>' if doc.tags else ''}
                        <div class="timeline-card-preview">"{preview}..."</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


def _type_badge(source_type: str) -> str:
    badges = {
        "pdf":      '<span class="badge badge-pdf">PDF</span>',
        "markdown": '<span class="badge badge-markdown">MD</span>',
        "url":      '<span class="badge badge-url">URL</span>',
        "notion":   '<span class="badge badge-notion">NOTION</span>',
    }
    return badges.get(source_type, '<span class="badge badge-markdown">DOC</span>')