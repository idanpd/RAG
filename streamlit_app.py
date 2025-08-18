# streamlit_app.py
import streamlit as st
from retriever import Retriever
from rag import build_prompt, answer_with_llm
from utils import load_config

cfg = load_config()
st.set_page_config(page_title="RAG File Search", layout="wide")
st.title("Local RAG File Search")

if "retriever" not in st.session_state:
    try:
        st.session_state.retriever = Retriever()
    except Exception as e:
        st.error(f"Failed to init retriever: {e}")

query = st.text_input("Enter your search or question")
if st.button("Search") or query:
    retriever = st.session_state.get("retriever")
    if not retriever:
        st.error("Retriever not ready.")
    else:
        with st.spinner("Searching..."):
            results = retriever.retrieve(query)
        st.subheader("Results")
        for r in results:
            st.markdown(f"**{r['path']}** — score: {r.get('cross_score', r['score']):.3f}")
            st.markdown(f"> {r['summary'][:300]}")
            if st.button(f"Show chunk {r['chunk_id']}", key=f"show_{r['chunk_id']}"):
                st.write(r['text'])
        if st.button("Ask LLM for Answer"):
            prompt = build_prompt(query, results)
            ans = answer_with_llm(prompt)
            st.subheader("LLM Answer")
            st.write(ans)
