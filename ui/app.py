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
    top_k = st.slider("Top K", min_value=1, max_value=20, value=6)
    columns = st.slider("Columns", min_value=1, max_value=6, value=3)
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


# --- Call the API -------------------------------------------------------------

try:
    response = requests.get(
        f"{api_base}/search",
        params={"q": query, "top_k": top_k},
        timeout=30,
    )
    response.raise_for_status()
except requests.RequestException as exc:
    st.error(f"Search failed: {exc}")
    st.stop()

data = response.json()
hits = data.get("hits", [])

if not hits:
    st.warning("No results. Did you index your photos with `python -m photo_search.indexer`?")
    st.stop()


# --- Render results as a grid -------------------------------------------------

st.subheader(f'Results for "{data["query"]}" — {len(hits)} hit(s)')

cols = st.columns(columns)
for i, hit in enumerate(hits):
    col = cols[i % columns]
    with col:
        photo_url = f"{api_base}/photo/{hit['id']}"
        st.image(photo_url, use_container_width=True)
        st.caption(f"**{hit['filename']}**  \nscore: `{hit['score']:.4f}`")
