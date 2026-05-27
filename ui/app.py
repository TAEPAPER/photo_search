"""Streamlit UI for photo search.

Run with:
    uv run streamlit run ui/app.py

Requires the FastAPI server to be running on http://localhost:8000.
Start it in another terminal:
    uv run uvicorn photo_search.api:app --port 8000
"""
from __future__ import annotations

import os

import requests
import streamlit as st

API_BASE = os.environ.get("PHOTO_SEARCH_API", "http://localhost:8000")

st.set_page_config(page_title="Photo Search", page_icon="🔎", layout="wide")
st.title("📸 Photo Search")
st.caption("Natural-language search over your photos, powered by CLIP + Qdrant.")


# --- Sidebar: settings --------------------------------------------------------

with st.sidebar:
    st.header("Settings")
    min_score = st.slider(
        "Minimum score",
        min_value=0.00, max_value=0.40, value=0.22, step=0.01,
        help="Cosine-similarity cutoff applied by Qdrant. "
             "CLIP scores are usually 0.20~0.30 for relevant matches; "
             "lower this slider to see more (noisier) results, raise it for stricter ones.",
    )
    columns = st.slider("Columns", min_value=1, max_value=6, value=3)
    auto_expand = st.checkbox(
        "Auto-expand query ('food' → 'a photo of food')",
        value=True,
        help="CLIP is trained on natural sentences. A one-word query often "
             "scores worse than 'a photo of <word>'.",
    )
    api_base = st.text_input("API base URL", value=API_BASE)

    st.divider()
    # Light health check so the user knows the server is up before they type.
    try:
        r = requests.get(f"{api_base}/health", timeout=2)
        if r.ok:
            st.success(f"API: {api_base}")
        else:
            st.error(f"API responded {r.status_code}")
    except requests.RequestException as exc:
        st.error(f"API unreachable: {exc}")


# --- Main: query box ----------------------------------------------------------

query = st.text_input(
    "Search",
    placeholder="e.g. a photo of food, mountains at sunset, a person smiling",
    label_visibility="collapsed",
)

if not query:
    st.info("Type a query above to search your photos.")
    st.stop()


# --- Build the actual query sent to the API -----------------------------------

# CLIP works better on short natural sentences than on bare nouns.
effective_query = (
    f"a photo of {query}"
    if auto_expand and not query.lower().startswith(("a photo", "a picture"))
    else query
)


# --- Call the API -------------------------------------------------------------

try:
    response = requests.get(
        f"{api_base}/search",
        params={"q": effective_query, "min_score": min_score},
        timeout=30,
    )
    response.raise_for_status()
except requests.RequestException as exc:
    st.error(f"Search failed: {exc}")
    st.stop()

data = response.json()
hits = data.get("hits", [])


# --- Render results as a grid -------------------------------------------------

st.subheader(
    f'Results for "{data["query"]}" — {len(hits)} hit(s) above score {min_score:.2f}'
)

if not hits:
    st.info(
        "No photos pass the score threshold. "
        "Try lowering it in the sidebar or rephrasing the query."
    )
    st.stop()

cols = st.columns(columns)
for i, hit in enumerate(hits):
    col = cols[i % columns]
    with col:
        photo_url = f"{api_base}/photo/{hit['id']}"
        st.image(photo_url, use_container_width=True)
        st.caption(f"**{hit['filename']}**  \nscore: `{hit['score']:.4f}`")
