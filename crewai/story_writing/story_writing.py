# Title: Story Writing with CrewAI and Ollama
#
# Description:
# This script demonstrates a multi-agent workflow: "Sequential Process".
# We have two specialized agents:
# 1. WRITER: Creates the initial draft.
# 2. EDITOR: Reviews and improves the draft.
#
# The output of the first task (Writing) is automatically passed as input
# to the second task (Editing), simulating a real-world content creation pipeline.
#
# Installation:
# pip install streamlit crewai langchain-community
#
# How to run:
# streamlit run story_writing.py

import streamlit as st
from crewai import Agent, Task, Crew
from langchain_community.llms import Ollama

st.title("Story Writing with CrewAI and Ollama")

# Input for story prompt
story_prompt = st.text_area(label="Enter a story prompt:")
button = st.button("Generate Story")

if button:
    if story_prompt:
        # --- Step 1: Define Specialized Agents ---
        
        # Initialize Ollama LLM
        llm = Ollama(model="llama3.1")

        # Agent 1: The Creative Writer
        writer = Agent(
            role="Writer",
            goal="Generate a creative story based on the given prompt.",
            backstory="This agent is a skilled writer with a vivid imagination. They will craft an engaging story using the provided prompt as inspiration.",
            llm=llm
        )

        # Agent 2: The Critical Editor
        editor = Agent(
            role="Editor",
            goal="Review and refine the generated story for clarity, coherence, and flow.",
            backstory="This agent is an experienced editor who will carefully review the story, provide feedback, and make necessary revisions to improve the overall quality.",
            llm=llm
        )

        # --- Step 2: Define Sequential Tasks ---

        # Task 1: Writing
        # The writer takes the user's prompt and produces the raw story.
        write_story = Task(
            description=f"Write a short story based on the prompt: {story_prompt}",
            expected_output="A creative short story with a beginning, middle, and end.",
            agent=writer
        )


        # Task 2: Editing
        # CRITICAL: In CrewAI, tasks are executed in order.
        # The Editor will automatically receive the output of the 'write_story' task
        # as its context, because they are in the same Crew.
        edit_story = Task(
            description="Review and edit the generated story to enhance clarity, coherence, and flow.",
            expected_output="The edited story with improved quality.",
            agent=editor
        )

        # --- Step 3: Form the Crew ---
        # The order in the 'tasks' list matters! [write_story, edit_story]
        story_crew = Crew(agents=[writer, editor], tasks=[write_story, edit_story])

        # --- Step 4: Execution ---
        # The result of kickoff() is the output of the FINAL task (the edited story).
        final_story = story_crew.kickoff()

        # --- Step 5: Display Results ---
        # We can also access the intermediate outputs (the draft) by accessing the Task object directly.

        # Create two columns layout in Streamlit
        col1, col2 = st.columns(2)

        # Column 1: Show the Writer's Draft
        with col1:
            st.header("First Story:")
            # .output contains the result of that specific task
            st.markdown(write_story.output)

        # Column 2: Show the Editor's Final Version
        with col2:
            st.header("Final Story:")
            st.markdown(final_story)