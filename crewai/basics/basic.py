# Compatibility shim for Python 3.14+: MUST be first!
# Some packages expect ast.NameConstant which was replaced by ast.Constant in
# newer Python versions. Provide a fallback so older packages (like crewai v0.11.2)
# that reference ast.NameConstant still work.
import ast
# Backwards-compatibility shims for old `ast` names removed/changed in newer
# Python versions. Map legacy nodes to `ast.Constant` so libraries that check
# `ast.NameConstant`, `ast.Num`, `ast.Str`, etc. continue to work.
for _name in ("NameConstant", "Num", "Str", "Bytes"):
    if not hasattr(ast, _name):
        setattr(ast, _name, ast.Constant)

# Title: Use Local Ollama LLM with CrewAI
#
# Description:
# This script introduces "CrewAI", a framework for orchestrating "AI Agents".
# Unlike a simple Chatbot (User <-> AI), CrewAI allows you to define:
# 1. AGENTS: AI personas with specific roles (e.g., Researcher, Writer).
# 2. TASKS: Specific jobs those agents must complete.
# 3. CREW: The team structure that manages how agents work together.
#
# This example creates a simple 1-Agent crew to demonstrate the syntax.
#
# Installation (Working versions for Python 3.14):
# PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 pip install crewai==0.11.2 streamlit langchain-community==0.0.38
#
# How to run:
# streamlit run basic.py
#
# Note: Make sure Ollama is running locally with the model you want to use
# Example: ollama run llama3.1

import streamlit as st

from crewai import Agent, Task, Crew  # The 3 core building blocks of CrewAI
from langchain_community.llms import Ollama

st.title("Use Local Ollama LLM with CrewAI")

# Input for the user's request
prompt = st.text_area(label="Enter your prompt:")
button = st.button("Generate")

if button:
    if prompt:
        # Initialize Ollama LLM
        ollama_llm = Ollama(model="llama3.1")

        # --- Step 1: Define the Agent ---
        # An Agent is an autonomous unit. It needs:
        # - role: What is its job title?
        # - goal: What is it trying to achieve?
        # - backstory: Context about its personality or expertise (helps the LLM roleplay).
        # - llm: Which model drives its brain? (LLM object, not a string)
        agent = Agent(
            role="Assistant",
            goal="Provide helpful responses based on user input.",
            backstory="This agent assists users by generating responses using a local Ollama LLM.",
            llm=ollama_llm
        )

        # --- Step 2: Define the Task ---
        # A Task is a specific unit of work. It needs:
        # - description: Instructions on what to do.
        # - expected_output: A definition of what the result should look like (text, list, etc.).
        # - agent: Which agent is responsible for this task?
        task = Task(
            description=f"Generate a response based on user input: {prompt}",
            expected_output="The generated response will be a flat text.",
            agent=agent  # Link the Task to the Agent
        )

        # --- Step 3: Define the Crew ---
        # The Crew is the container that holds agents and tasks together.
        # It manages the workflow (e.g., sequential execution).
        crew = Crew(agents=[agent], tasks=[task])

        # --- Step 4: Kickoff ---
        # Start the process. The agents will begin working on their tasks.
        # This returns the final output of the last task.
        result = crew.kickoff()

        # Display the result
        st.markdown(result)