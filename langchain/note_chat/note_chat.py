# Title: Chat with Your Note
#
# Description:
# This is a "Personal Knowledge Base" application.
# It allows the user to:
# 1. SAVE snippets of text (notes) into a permanent storage file (`note.text`).
# 2. ASK questions about anything they have previously saved.
#
# It uses the same RAG (Retrieval-Augmented Generation) logic as the previous lesson,
# but adds the ability to accumulate data over time rather than pasting it all at once.
#
# Installation:
# pip install streamlit==1.33.0 langchain==0.2.16 langchain-community==0.2.17 spacy==3.7.4
# python -m spacy download en_core_web_md
#
# How to run:
# streamlit run note_chat.py
#
# Note: Ensure you create an empty file named "note.text" in the folder before running.

import streamlit as st
from langchain.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import spacy

# Load SpaCy for embedding calculations
nlp = spacy.load("en_core_web_md")

st.title("Chat with Your Note")

# --- Part 1: Adding New Notes ---
# We use st.form to group the text area and submit button.
# clear_on_submit=True wipes the text box after saving, improving UX.
with st.form(key='my_form', clear_on_submit=True):
    # Input fields for the form
    note = st.text_area(label="Paste your note here.", height=300)
    # Submit button
    submit_button = st.form_submit_button(label='Save')

if submit_button:
    if note:
        # Open the persistence file in read mode ('r') to check existing content
        with open("note.text", 'r') as file:
            content = file.read()

            # Simple deduplication: Check if this exact note already exists
            if note not in content:
                # Append the new note with some spacing
                content += f"\n\n{note}"

                # Re-open the file in write mode ('w') to save the updated content.
                # Note: 'w' overwrites the file, so we write back the FULL content (old + new).
                with open("note.text", 'w') as file:
                    file.write(content)

# --- Part 2: Q&A Interface ---
question = st.text_input(label="Enter your question:")
button = st.button("ASK")

if button:
    if question:
        # Initialize the local LLM
        llm = Ollama(model='llama3.1')  # Specify your model here

        # Define the RAG prompt template
        template = """You are a helpful assistant. Please answer the following question based on the below text: Question: {question} Text: {text}
        """
        prompt_template = PromptTemplate(template=template, input_variables=["text", "question"])

        # Load the accumulated notes from the file
        with open("note.text", 'r') as file:
            content = file.read()

        # --- RAG Workflow (Same as Lesson 21) ---

        # 1. Chunking: Split the massive note file into smaller pieces
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.create_documents([content])

        # 2. Semantic Search: Find chunks relevant to the user's question
        similarities = []
        for chunk in chunks:
            similarity_score = nlp(question).similarity(nlp(chunk.page_content))
            similarities.append((similarity_score, chunk.page_content))

        # 3. Filtering: Pick the top 3 matches
        ordered_chunks = sorted(similarities, key=lambda x: x[0], reverse=True)[:5]

        # 4. Context Construction: Combine the top matches into a string
        text = ""
        for score, sentence in ordered_chunks:
            # Show the user what data was retrieved
            st.markdown(f"Similarity: {score:.4f} - Sentence: {sentence}")
            text += f"{sentence}\n\n"

        # 5. Generation: Get the answer from the LLM
        chain = LLMChain(llm=llm, prompt=prompt_template)
        answer = chain.run({"text": text,
                            "question": question})

        st.subheader("Answer:")
        st.markdown(answer)