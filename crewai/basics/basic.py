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
# Installation:
# pip install streamlit==1.33.0 crewai langchain-ollama
# Worked these versions on linux
# pip install streamlit==1.33.0 crewai==0.64.0 langchain-community==0.2.17
#
# How to run:
# streamlit run basic.py

import streamlit as st
from crewai import Agent, Task, Crew  # The 3 core building blocks of CrewAI

st.title("Use Local Ollama LLM with CrewAI")

# Input for the user's request
prompt = st.text_area(label="Enter your prompt:")
button = st.button("Generate")

if button:
    if prompt:
        # --- Step 1: Define the Agent ---
        # An Agent is an autonomous unit. It needs:
        # - role: What is its job title?
        # - goal: What is it trying to achieve?
        # - backstory: Context about its personality or expertise (helps the LLM roleplay).
        # - llm: Which model drives its brain? (LLM object, not a string)
        # llm = Ollama(model="llama3.1") // For Linux Local Run
        agent = Agent(
            role="Assistant",
            goal="Provide helpful responses based on user input.",
            backstory="This agent assists users by generating responses using a local Ollama LLM.",
            llm="ollama/llama3.1",
            # llm = llm // For linux local run
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