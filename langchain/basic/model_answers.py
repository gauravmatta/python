# Title: Get Model Answer Questions Based on Long Stories
#
# Description:
# This is a full "RAG" (Retrieval-Augmented Generation) system built from scratch!
# The problem: We have a very long text (too big for the LLM). We want to ask a question.
# The solution:
# 1. SPLIT the text into chunks.
# 2. SEARCH: Compare the user's question to every chunk (using SpaCy embeddings).
# 3. FILTER: Pick only the top 3 most relevant chunks.
# 4. ANSWER: Send ONLY those 3 relevant chunks to Llama 3 to get the final answer.
#
# This saves money/compute and improves accuracy by removing irrelevant noise.
#
# Installation:
# pip install streamlit==1.33.0 langchain==0.2.16 langchain-community==0.2.17 spacy==3.7.4
# python -m spacy download en_core_web_md
#
# How to run:
# streamlit run model_answers.py

import streamlit as st
from langchain.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import spacy

# Load the SpaCy model for vector embeddings
# Ensure you run: python -m spacy download en_core_web_md
nlp = spacy.load("en_core_web_md")

st.title("Text Summarizer with Chunking and Similarity")

# Input for the long text (Source Material)
long_text = st.text_area(label="Paste your long text here.", height=300)

# Input for the User's Question
question = st.text_input(label="Enter your question:")
button = st.button("Answer")

# Optional: Show character count
st.markdown(f"{len(long_text)}")

if button:
    if long_text:
        # Initialize the local LLM
        llm = Ollama(model='llama3.1')  # Specify your model here

        # Define the prompt template
        # The AI will see the "Text" variable (which we will fill with only relevant chunks)
        # and answer the "Question".
        template = """
        You are a helpful assistant. Please answer the following question based on the below text: Question: {question} Text: {text}
        """
        prompt_template = PromptTemplate(template=template, input_variables=["text", "question"])

        # --- Step 1: Chunking ---
        # Split the massive text into manageable pieces (approx 1000 characters each).
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.create_documents([long_text])

        # --- Step 2: Semantic Search ---
        # We need to find which chunks actually contain the answer to the user's question.
        similarities = []

        for chunk in chunks:
            # Calculate similarity between the USER'S QUESTION and the CHUNK content.
            # If the question is about "Apples", a chunk discussing "Apples" will have a high score.
            similarity_score = nlp(question).similarity(nlp(chunk.page_content))
            similarities.append((similarity_score, chunk.page_content))

        # --- Step 3: Filter / Retrieve ---
        # Sort by score (highest first) and take the top 3 results.
        # This is the "Retrieval" part of RAG.
        ordered_chunks = sorted(similarities, key=lambda x: x[0], reverse=True)[:5]

        # Prepare the context string
        text = ""
        for score, sentence in ordered_chunks:
            # Show the user which parts were selected (Transparency)
            st.markdown(f"Similarity: {score:.4f} - Sentence: {sentence}")
            # Concatenate the top 3 chunks into one context block
            text += f"{sentence}\n\n"

        # --- Step 4: Generation ---
        # Send the filtered context + question to the LLM.
        chain = LLMChain(llm=llm, prompt=prompt_template)
        answer = chain.run({"text": text,
                            "question": question})

        st.subheader("Answer:")
        st.markdown(answer)