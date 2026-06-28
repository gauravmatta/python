# Title: Use PromptTemplate and LLMChain
#
# Description:
# This script introduces two core concepts of LangChain:
# 1. PromptTemplate: A way to create dynamic, reusable prompts where you can insert variables (like the user's question).
# 2. LLMChain: A simple "Chain" that links a PromptTemplate and an LLM together.
#
# Instead of just sending the raw user input to the model, we wrap it in a structure
# (e.g., "You are a helpful assistant...") to control the AI's behavior and formatting.
#
# Installation:
# pip install streamlit==1.33.0 langchain==0.2.16 langchain-community==0.2.17
#
# How to run:
# streamlit run prompt_template.py

import streamlit as st
from langchain.llms import Ollama  # Wrapper for the local model
from langchain.chains import LLMChain  # The simplest chain: Prompt -> LLM
from langchain.prompts import PromptTemplate  # For creating flexible prompt strings

st.title("Local LLM with Langchain!")

# Input for the prompt
prompt = st.text_area(label="Write your prompt.")
button = st.button("Okay")

if button:
    if prompt:
        # Initialize the local LLM
        llm = Ollama(model='llama3.1')  # Specify your model here

        # --- Define the Prompt Template ---
        # A template is a string with placeholders (variables inside curly braces {}).
        # This allows us to "frame" the user's question.
        # Here, we force the AI to answer as a "list with short items" regardless of what the user asks.
        template = """You are a helpful assistant. You have been asked the following question:
        Question: {question}
        Please provide a detailed and thoughtful response as a list with short items.
        """
        # Create the PromptTemplate object.
        # input_variables=["question"] tells LangChain that we will provide a value for {question} later.
        prompt_template = PromptTemplate(template=template, input_variables=["question"])

        # --- Create the LLMChain ---
        # An LLMChain combines the PromptTemplate and the LLM into a single object.
        # When we run this chain, it will:
        # 1. Take our input (the user's prompt).
        # 2. Fill it into the {question} slot of the template.
        # 3. Send the complete, formatted string to the LLM.
        chain = LLMChain(llm=llm, prompt=prompt_template)

        # Generate a response using the LLMChain.
        # chain.run(prompt) takes the user's input string, assigns it to the input variable,
        # and executes the chain.
        response = chain.run(prompt)

        # Display the response
        st.markdown(response)