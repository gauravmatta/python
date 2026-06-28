# Title: Learn Chunking Long Texts
#
# Description:
# This script tackles a common problem: "Context Window Limits".
# LLMs cannot read infinite amounts of text at once. If you try to summarize a whole book,
# the model will crash or cut off the text.
#
# The solution is "Chunking":
# 1. Split the long text into smaller pieces (chunks).
# 2. Summarize each chunk individually.
# 3. Combine those small summaries into one final summary (Map-Reduce approach).
#
# Installation:
# pip install streamlit==1.33.0 langchain==0.2.16 langchain-community==0.2.17
#
# How to run:
# streamlit run chunking.py

import streamlit as st
from langchain.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Tool for splitting text intelligently

st.title("Text Summarizer with Chunking")

# Input for the long text (e.g., a long article or essay)
long_text = st.text_area(label="Paste your long text here.", height=300)
button = st.button("Summarize")

if button:
    if long_text:
        # Initialize the local LLM
        llm = Ollama(model='llama3.1')  # Specify your model here

        # Define the prompt template for summarization
        template = """You are a helpful assistant. Please summarize the following text:
        Text: {text} Provide a concise summary.
        """
        prompt_template = PromptTemplate(template=template, input_variables=["text"])

        # --- Chunking Strategy ---
        # RecursiveCharacterTextSplitter tries to split text by paragraphs, then sentences, then words.
        # chunk_size=1000: Each chunk will be roughly 1000 characters long.
        # chunk_overlap=150: The end of chunk 1 will repeat at the start of chunk 2.
        # Overlap is crucial so the AI doesn't lose context if a sentence is cut in half between chunks.
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

        # Create the chunks from the user's text
        chunks = text_splitter.create_documents([long_text])
        st.markdown(chunks) # Optional: verify the chunks

        # Initialize an empty list to store the summaries of each part
        summaries = []

        # --- Phase 1: Map (Summarize each chunk) ---
        for chunk in chunks:
            # Create the LLMChain for each chunk (reusing the same template)
            chain = LLMChain(llm=llm, prompt=prompt_template)

            # Show the user which part is currently being processed
            st.markdown(chunk.page_content)

            # Generate a summary for this specific chunk
            summary = chain.run(chunk.page_content)
            summaries.append(summary)

        # --- Phase 2: Reduce (Summarize the summaries) ---
        # Join all the mini-summaries into one text block.
        final_summary = "\n\n".join(summaries)

        # Display the intermediate combined summary
        st.markdown(final_summary)

        # Ask the LLM one last time to condense everything into a final polished summary.
        # Note: We are calling 'llm()' directly here for simplicity, but you could use a Chain too.
        combined_final_summary = llm(f"Combine the following summaries into one concise summary: \n\n{final_summary}")

        # Display the final result
        st.subheader("Final Summary:")
        st.markdown(combined_final_summary)