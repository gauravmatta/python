# Title: Personal Finance Advisor
#
# Description:
# This script applies the Multi-Agent framework to the Finance domain.
# It creates a team of two experts:
# 1. Financial Planner: Focuses on SAVING (budgeting, cutting costs, timelines).
# 2. Investment Advisor: Focuses on GROWING (stocks, bonds, risk tolerance).
#
# By splitting "Finance" into "Saving" vs "Investing", we get a holistic
# financial strategy that covers both defense (saving) and offense (investing).
#
# Installation:
# pip install streamlit crewai langchain-community
#
# How to run:
# streamlit run finance_advisor.py

import streamlit as st
from crewai import Agent, Task, Crew
from langchain_community.llms.ollama import Ollama

st.title("Personal Finance Advisor")

# Input for the user's specific goal (e.g., "Retire by 40", "Buy a Tesla")
financial_goal = st.text_input(label="Enter your financial goal (e.g., saving for a house):")
button = st.button("Get Advice")

if button:
    if financial_goal:
        # --- Step 1: Define Specialized Agents ---

        llm = Ollama(model="llama3.1")

        # Agent 1: The Saver
        planner = Agent(
            role="Financial Planner",
            goal="Create a savings plan to achieve the financial goal.",
            backstory="This agent helps users manage their finances effectively, focusing on budgeting and safe accumulation.",
            llm=llm
        )

        # Agent 2: The Investor
        # This agent takes over where the saver leaves off, suggesting what to do with the saved money.
        investor = Agent(
            role="Investment Advisor",
            goal="Provide investment options based on user’s risk tolerance.",
            backstory="This agent specializes in finding suitable investment opportunities to maximize returns.",
            llm=llm
        )

        # --- Step 2: Define Tasks ---

        # Task 1: The Plan
        # Output: A step-by-step guide on how to accumulate the necessary funds.
        create_plan = Task(
            description=f"Develop a savings plan for achieving: {financial_goal}.",
            expected_output="A detailed savings plan with steps to follow.",
            agent=planner
        )

        # Task 2: The Growth
        # Note: The Investor implicitly uses the context of the 'Savings Plan' to recommend appropriate vehicles.
        # (e.g., if the plan is short-term, the investor shouldn't recommend volatile stocks).
        suggest_investments = Task(
            description=f"Suggest investment options based on user's financial situation.",
            expected_output="A list of recommended investments.",
            agent=investor
        )

        # --- Step 3: Execution ---
        # Run the Planner first, then the Investor.
        finance_crew = Crew(agents=[planner, investor], tasks=[create_plan, suggest_investments])

        # Kickoff the workflow
        final_advice = finance_crew.kickoff()

        # --- Step 4: Display Results ---

        # Show the "Defense" (Savings)
        st.subheader("Savings Plan:")
        st.markdown(create_plan.output)

        # Show the "Offense" (Investments)
        st.subheader("Investment Options:")
        st.markdown(final_advice)
