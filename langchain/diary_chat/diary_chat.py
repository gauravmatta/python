# Title: Chat with Your Diary
#
# Description:
# This is an advanced variation of the "Chat with Note" app.
# Instead of storing everything in a messy text file, we use a STRUCTURED data format: JSON.
#
# Key Features:
# 1. Date-Based Storage: Notes are saved with the date as the key (e.g., "2023-10-27": "Went to the park").
# 2. Structured Retrieval: We can access specific days easily.
# 3. JSON Persistence: Using Python's `json` library to save/load the dictionary to a file.
#
# Installation:
# pip install streamlit==1.33.0 langchain==0.2.16 langchain-community==0.2.17 spacy==3.7.4
# python -m spacy download en_core_web_md
#
# How to run:
# streamlit run diary_chat.py
#
# Note: Ensure you have a file named "diary.json" containing at least "{}" (empty JSON object)
# in the folder before running, or the script might error on first load.

import streamlit as st
from langchain.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import spacy
from datetime import datetime
import json

# Load SpaCy for semantic search
nlp = spacy.load("en_core_web_md")

st.title("Chat with Your Diary")

# --- UI: Date & Note Input ---
# Get today's date automatically
today = datetime.now().date()

# Allow the user to pick a date (defaults to today)
selected_date = st.date_input("Select a date", value=today)

with st.form(key='my_form', clear_on_submit=True):
    note = st.text_area(label="Write your diary here.", height=300)
    submit_button = st.form_submit_button(label='Save')

if submit_button:
    if note:
        # Convert date object to string (JSON keys must be strings)
        selected_date = str(selected_date)

        # Load existing diary data
        # Mode 'r' reads the JSON structure into a Python dictionary
        with open("diary.json", 'r') as json_file:
            data = json.load(json_file)

            # Check if an entry already exists for this date
            if selected_date in data.keys():
                # Append new note if it's not a duplicate
                if note not in data[selected_date]:
                    data[selected_date] += f"\n\n{note}"

                    # Save back to file
                    with open("diary.json", 'w') as json_file:
                        json.dump(data, json_file)  # .dump() writes the dictionary back as JSON text
            else:
                # Create a new entry for this date
                data[selected_date] = note

                # Save back to file
                with open("diary.json", 'w') as json_file:
                    json.dump(data, json_file)

# --- Q&A Section ---
question = st.text_input(label="Enter your question:")
button = st.button("ASK")

if button:
    if question:
        # Initialize LLM
        llm = Ollama(model='llama3.1')

        # Define RAG Prompt
        # Notice we tell the AI it is answering based on a "Diary".
        template = """You are a helpful assistant. Please answer the following question based on the below diary: Question: {question} Diary: {text}
        """
        prompt_template = PromptTemplate(template=template, input_variables=["text", "question"])

        # Load the entire diary
        with open("diary.json", 'r') as json_file:
            data = json.load(json_file)

        # --- RAG Logic for Structured Data ---
        # Instead of chunking a raw text file, we iterate through the dictionary items.
        # Each "Chunk" is effectively one Day's entry.
        similarities = []

        # data.items() gives us tuples like: ("2023-10-27", "Today I went to...")
        for date, diary in data.items():
            # Calculate similarity between the Question and the Diary Entry
            similarity_score = nlp(question).similarity(nlp(diary))

            # We construct a context string that explicitly includes the Date.
            # This helps the AI answer time-based questions like "What did I do last Friday?"
            similarities.append((similarity_score, f"Date: {date}\nDiary: {diary}"))

        # Sort and pick top 3 relevant days
        ordered_chunks = sorted(similarities, key=lambda x: x[0], reverse=True)[:5]

        text = ""
        for score, sentence in ordered_chunks:
            # Display retrieved entries
            st.markdown(f"Similarity: {score:.4f} - Sentence: {sentence}")
            text += f"{sentence}\n\n"

        # Generate Answer
        chain = LLMChain(llm=llm, prompt=prompt_template)
        answer = chain.run({"text": text,
                            "question": question})

        st.subheader("Answer:")
        st.markdown(answer)