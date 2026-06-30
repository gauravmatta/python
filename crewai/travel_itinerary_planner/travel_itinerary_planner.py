# Title: Travel Itinerary Planner
#
# Description:
# This script applies the Multi-Agent concept to the Travel domain.
# It creates a "Travel Agency" team:
# 1. Planner: Focuses on logistics, schedule, and structure.
# 2. Local Expert: Focuses on culture, hidden gems, and specific recommendations.
#
# This shows how breaking a complex problem (planning a vacation) into
# distinct perspectives (Logistics vs. Experience) yields a richer result.
#
# Installation:
# pip install streamlit crewai langchain-community
#
# How to run:
# streamlit run travel_itinerary_planner.py

import streamlit as st
from crewai import Agent, Task, Crew
from langchain_community.llms import Ollama

st.title("Travel Itinerary Planner")

# Input for travel destination (e.g., "Tokyo, Japan")
destination = st.text_input(label="Enter your travel destination:")
button = st.button("Plan Trip")

if button:
    if destination:
        # --- Step 1: Define Specialized Agents ---
        
        # Initialize Ollama LLM
        llm = Ollama(model="llama3.1")

        # Agent 1: The Logistician
        planner = Agent(
            role="Trip Planner",
            goal="Create a detailed travel itinerary based on the destination.",
            backstory="This agent specializes in planning exciting trips.",
            llm=llm
        )

        # Agent 2: The Local Guide
        local_expert = Agent(
            role="Local Expert",
            goal="Provide insights and recommendations about the destination.",
            backstory="This agent knows all the best spots in town!",
            llm=llm
        )

        # --- Step 2: Define Tasks ---

        # Task 1: Structure the Trip
        create_itinerary = Task(
            description=f"Plan a trip itinerary for {destination}.",
            expected_output="A detailed itinerary including activities and timings.",
            agent=planner
        )

        # Task 2: Add Color/Context
        # Note: CrewAI passes the context of previous tasks automatically.
        # The Local Expert will likely see the Planner's itinerary and can add
        # specific tips relevant to those locations.
        get_local_insights = Task(
            description=f"Provide local insights about {destination}.",
            expected_output="Recommendations for places to visit and eat.",
            agent=local_expert
        )

        # --- Step 3: Execution ---
        # The planner runs first to set the skeleton of the trip.
        # The expert runs second to fill in the details.
        trip_crew = Crew(agents=[planner, local_expert], tasks=[create_itinerary, get_local_insights])

        # Start the process
        final_itinerary = trip_crew.kickoff()

        # --- Step 4: Display Results ---

        st.subheader("Planned Itinerary:")
        st.markdown(create_itinerary.output)  # Output from the Planner

        st.subheader("Local Insights:")
        st.markdown(final_itinerary)  # Output from the Expert