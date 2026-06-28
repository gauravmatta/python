import streamlit as st

st.title("Ollama!")
prompt = st.text_area("Enter your prompt here:")
if st.button("Okay"):
    if prompt:
        st.markdown(prompt)