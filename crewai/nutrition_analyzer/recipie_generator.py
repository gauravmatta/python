# Title: Recipe Generator and Nutrition Analyzer
#
# Description:
# This script reinforces the "Sequential Workflow" concept with a new domain: Cooking.
# 1. Agent 1 (Chef): Creates something new (a recipe).
# 2. Agent 2 (Nutritionist): Analyzes the output of Agent 1.
#
# This pattern (Creator -> Analyst) is extremely common in AI applications.
#
# Installation:
# pip install streamlit crewai langchain-community
#
# How to run:
# streamlit recipie_generator.py

import streamlit as st
from crewai import Agent, Task, Crew
from langchain_community.llms import Ollama
from langchain.tools import tool

st.title("Recipe Generator and Nutrition Analyzer")

# Define a simple tool for agents to communicate
@tool
def ask_question_to_co_worker(question: str, coworker: str = None) -> str:
    """Ask a question to a coworker (for inter-agent communication)."""
    if coworker:
        return f"Question asked to {coworker}: {question}"
    return f"Question asked: {question}"

# Input for recipe idea (e.g., "A high-protein vegan breakfast")
recipe_prompt = st.text_area(label="Enter a recipe idea:")
button = st.button("Generate Recipe")

if button:
    if recipe_prompt:
        # --- Step 1: Define Agents (Creator & Analyst) ---
        
        # Initialize Ollama LLM
        llm = Ollama(model="llama3.1")

        # Agent 1: The Creator
        chef = Agent(
            role="Chef",
            goal="Generate a creative recipe based on the given idea.",
            backstory="This agent is a skilled chef who loves to create delicious recipes.",
            llm=llm
        )

        # Agent 2: The Analyst
        # This agent needs to understand food science to critique the Chef's work.
        nutritionist = Agent(
            role="Nutritionist",
            goal="Analyze the generated recipe for nutritional value.",
            backstory="This agent is a nutrition expert who evaluates recipes for health benefits.",
            llm=llm
        )

        # --- Step 2: Define Tasks ---

        # Task 1: Generation
        generate_recipe = Task(
            description=f"Create a recipe based on the idea: {recipe_prompt}",
            expected_output="A detailed recipe with ingredients and instructions.",
            agent=chef
        )

        # Task 2: Analysis
        # The Nutritionist implicitly receives the Chef's output as context.
        analyze_nutrition = Task(
            description="Evaluate the nutritional value of the generated recipe.",
            expected_output="Nutritional analysis of the recipe.",
            agent=nutritionist
        )

        # --- Step 3: Execution ---
        # The Crew ensures Task 1 finishes before Task 2 starts.
        recipe_crew = Crew(agents=[chef, nutritionist], tasks=[generate_recipe, analyze_nutrition])

        # Kickoff returns the result of the LAST task (the nutrition analysis)
        final_recipe = recipe_crew.kickoff()

        # --- Step 4: Display Both Results ---
        # We want to see the Recipe AND the Analysis, so we print both.

        st.subheader("Generated Recipe:")
        st.markdown(generate_recipe.output)  # Output from Task 1

        st.subheader("Nutritional Analysis:")
        st.markdown(final_recipe)  # Output from Task 2 (Crew result)