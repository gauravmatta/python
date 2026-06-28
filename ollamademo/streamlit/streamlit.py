import ollama

import streamlit as st

st.title("Ollama!")
prompt = st.text_area("Enter your prompt here:")
if st.button("Okay"):
    if prompt:
        st.markdown(prompt)
        response = ollama.generate('gemma3', prompt=prompt)
        st.markdown(response['response'])