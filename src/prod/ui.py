from __future__ import annotations

import os
import sys

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Ensure the project root is on the path so `src.*` imports resolve.
# ui.py lives in <root>/src/prod/ui.py => 3 dirname() calls to reach the root.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import Config  # noqa: E402 - requires PROJECT_ROOT on sys.path

BACKEND_URL = Config.BACKEND_URL.rstrip("/")

st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="wide")

st.title("🤖 AI Assistant")
st.caption("Production AI assistant with RAG, tool calling, and structured output.")


def sidebar():
    with st.sidebar:
        st.header("Settings")
        temperature = st.slider("Temperature", 0.0, 2.0, Config.TEMPERATURE, 0.1)
        use_cache = st.checkbox("Use response cache", value=True)
        st.divider()
        st.subheader("Backend Status")
        try:
            resp = requests.get(f"{BACKEND_URL}/status", timeout=5)
            data = resp.json()
            st.success("✅ Backend online")
            st.write(f"Provider: {data.get('provider')}")
            st.write(f"Model: {data.get('model')}")
            st.write(f"Vector chunks: {data.get('vector_store_chunks')}")
            st.write(f"Cache entries: {data.get('cache_size')}")
        except Exception:
            st.error(f"❌ Backend unreachable at {BACKEND_URL}")
        st.divider()
        st.subheader("Ingest Documents")
        ingest_path = st.text_input("File/dir path")
        if st.button("Ingest", type="primary"):
            try:
                resp = requests.post(f"{BACKEND_URL}/ingest", json={"source": ingest_path}, timeout=300)
                if resp.status_code == 200:
                    st.success(f"Ingested {resp.json()['ingested']} chunks")
                else:
                    st.error(f"Error: {resp.json().get('detail', 'Unknown')}")
            except Exception as e:
                st.error(f"Error: {e}")
        return temperature, use_cache


def main():
    temperature, use_cache = sidebar()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    structured = st.toggle("Structured JSON output", value=False)

    prompt = st.chat_input("Ask the AI assistant...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        payload = {
            "message": prompt,
            "structured": structured,
            "temperature": temperature,
            "use_cache": use_cache,
        }

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    resp = requests.post(f"{BACKEND_URL}/chat", json=payload, timeout=120)
                    if resp.status_code == 200:
                        data = resp.json()
                        cache_hint = " *(cached)*" if data.get("cached") else ""
                        st.markdown(data["response"] + cache_hint)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": data["response"] + cache_hint}
                        )
                    else:
                        detail = resp.json().get("detail", "Unknown error")
                        st.error(f"Backend error ({resp.status_code}): {detail}")
                except Exception as e:
                    st.error(f"Failed to reach backend: {e}")


if __name__ == "__main__":
    main()
