"""Ingest page — add documents to the knowledge base."""

import tempfile
import os
import streamlit as st


def render_ingest(embedding_backend: str = "local"):

    st.markdown('<div class="brand">Add knowledge</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">Ingest documents into your second brain</div>', unsafe_allow_html=True)

    # ── Session state for queued sources ───────────────────────────────
    if "queued_sources" not in st.session_state:
        st.session_state.queued_sources = []
    if "tmp_dir" not in st.session_state:
        st.session_state.tmp_dir = tempfile.mkdtemp()

    # ── Input tabs ─────────────────────────────────────────────────────
    tab_url, tab_file, tab_text = st.tabs(["🌐  Web URL", "📄  Upload file", "✏️  Paste text"])

    # URL tab
    with tab_url:
        st.markdown("Add any article, blog post, or documentation page.")
        url_input = st.text_area(
            "URLs (one per line)",
            placeholder="https://example.com/article",
            height=120,
            label_visibility="collapsed",
        )
        if st.button("Add URLs"):
            urls = [u.strip() for u in url_input.strip().splitlines() if u.strip()]
            if urls:
                for u in urls:
                    if u not in st.session_state.queued_sources:
                        st.session_state.queued_sources.append(u)
                st.success(f"Added {len(urls)} URL(s) to queue")
            else:
                st.warning("Enter at least one URL.")

    # File upload tab
    with tab_file:
        st.markdown("Upload PDFs or Markdown files.")
        uploaded = st.file_uploader(
            "Files",
            type=["pdf", "md", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if st.button("Add files"):
            if uploaded:
                added = 0
                for f in uploaded:
                    path = os.path.join(st.session_state.tmp_dir, f.name)
                    with open(path, "wb") as out:
                        out.write(f.read())
                    if path not in st.session_state.queued_sources:
                        st.session_state.queued_sources.append(path)
                        added += 1
                st.success(f"Added {added} file(s) to queue")
            else:
                st.warning("Please upload at least one file first.")

    # Paste text tab
    with tab_text:
        st.markdown("Paste any text — meeting notes, ideas, snippets.")
        col1, col2 = st.columns(2)
        with col1:
            note_title = st.text_input("Title", placeholder="My note title")
        with col2:
            note_tags = st.text_input("Tags (comma separated)", placeholder="rag, research")
        note_text = st.text_area(
            "Content",
            placeholder="Paste your text here…",
            height=200,
            label_visibility="collapsed",
        )
        if st.button("Add note"):
            if note_text.strip() and note_title.strip():
                tags  = ", ".join(f'"{t.strip()}"' for t in note_tags.split(",") if t.strip())
                fname = note_title.lower().replace(" ", "_") + ".md"
                path  = os.path.join(st.session_state.tmp_dir, fname)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(f"---\ntitle: \"{note_title}\"\ntags: [{tags}]\n---\n\n{note_text}")
                if path not in st.session_state.queued_sources:
                    st.session_state.queued_sources.append(path)
                st.success(f"Note '{note_title}' added to queue")
            else:
                st.warning("Please enter both a title and content.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Queue preview ──────────────────────────────────────────────────
    if st.session_state.queued_sources:
        st.markdown(f"**{len(st.session_state.queued_sources)} source(s) ready to index:**")
        for s in st.session_state.queued_sources:
            name = s.split("/")[-1].split("\\")[-1]
            st.markdown(f"- `{name}`")

        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button("Build / update index", type="primary"):
                _run_indexing(
                    st.session_state.queued_sources,
                    embedding_backend,
                )
        with col2:
            if st.button("Clear queue"):
                st.session_state.queued_sources = []
                st.rerun()
    else:
        st.markdown("**No sources queued yet.** Add URLs, files, or notes above.")
        st.button("Build / update index", disabled=True)

    # ── Current docs list ──────────────────────────────────────────────
    if st.session_state.docs:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(f"**{len(st.session_state.docs)} documents currently indexed**")
        for doc in sorted(st.session_state.docs, key=lambda d: d.created_at, reverse=True):
            st.markdown(f"""
            <div class="source-card" style="margin-bottom:0.4rem;">
                <div class="source-title">{doc.title}</div>
                <div style="font-family: DM Mono, monospace; font-size: 0.65rem; color: #aaa;">
                    {doc.source_type.value.upper()} · {doc.word_count} words · {doc.source_uri}
                </div>
            </div>
            """, unsafe_allow_html=True)


def _run_indexing(sources, embedding_backend):
    """Run the full ingestion + indexing pipeline."""
    from ingestion.pipeline import IngestionPipeline
    from week5_pipeline import Week5Pipeline

    with st.status("Building your knowledge base…", expanded=True) as status:

        st.write(f"Ingesting {len(sources)} source(s)…")
        ingestor = IngestionPipeline()
        docs     = ingestor.run(sources)

        if not docs:
            st.error("No documents could be ingested. Check your sources.")
            status.update(label="Failed", state="error")
            return

        st.write(f"Chunking and embedding {len(docs)} document(s)…")
        pipeline = Week5Pipeline(
            embedding_backend=embedding_backend,
            store_mode="memory",
        )
        pipeline.index(docs)

        st.session_state.pipeline        = pipeline
        st.session_state.docs            = docs
        st.session_state.indexed         = True
        st.session_state.queued_sources  = []   # clear queue after success

        status.update(label="Knowledge base ready!", state="complete")

    st.success(f"Indexed {len(docs)} docs · {pipeline.vector_store.count()} chunks")
    st.balloons()