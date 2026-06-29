# Title: Get Embedding and Similarity Between Two Texts
#
# Description:
# This script introduces "Vector Embeddings" and "Semantic Similarity".
# Computers cannot understand text meaning directly. They understand numbers.
# "Embeddings" turn text into lists of numbers (vectors) where similar meanings are close together.
#
# This script uses the SpaCy library to:
# 1. Turn two blocks of text into vectors.
# 2. Calculate how similar they are (a score from 0 to 1).
# 3. Benchmark how fast this calculation is.
#
# Installation:
# 1. Install SpaCy and NumPy:
#    pip install spacy==3.7.4 numpy
#
# 2. Download the English language model (Medium or Large size is required for vectors!):
#    python -m spacy download en_core_web_md
#
# Note: The "sm" (small) model does NOT include word vectors, so similarity won't work well.
# You must use "md" (medium) or "lg" (large).
#
# How to run:
# python text_similarity

import time

import numpy as np
import spacy.cli

# Load the SpaCy model
# This loads the dictionary and the vector embeddings into memory.
# Ensure you have run: python -m spacy download en_core_web_md
nlp = spacy.load("en_core_web_md")


# --- Function: Calculate Similarity ---
# This function takes two strings, converts them to vector representations,
# and computes the cosine similarity between them.
def calculate_similarity(text1, text2):
    # nlp(text1) processes the text and creates a Doc object.
    # The .similarity() method compares the vector of Doc 1 with Doc 2.
    # Under the hood, it performs a Dot Product divided by the magnitudes (Cosine Similarity).
    #
    # Note: The commented-out code shows the manual math (Vector1 @ Vector2 / Norms)
    # which is what .similarity() does internally.
    similarity = nlp(text1).similarity(nlp(text2))

    return similarity

def calculate_similarity_vector(text1, text2):
    vector1 = nlp(text1).vector
    vector2 = nlp(text2).vector
    similarity = vector1 @ vector2/(np.linalg.norm(vector1)*np.linalg.norm(vector2))

    return similarity

# --- Input Data ---
# Two paragraphs of scientific text about "DNN models" and "airfoils".
# Since they are about the same specific topic, we expect a high similarity score.
text1 = """The findings reveal an impressive average accuracy of 99% between the CL values obtained 
from analytical methods and the DNN model. This high level of agreement demonstrates the 
DNN model's exceptional capability in accurately predicting and designing NACA 4-digit 
airfoils.
"""
text2 = """To further validate the DNN model's performance, we designed 10,000 airfoils using both 
analytical methods and the DNN model. The results show that the DNN model can design 
airfoils four times faster than the analytical method. This significant time advantage highlights 
the computational efficiency of the DNN approach, which is crucial in practical applications 
where rapid design iterations are necessary. """

# --- Benchmark Speed ---
# We use the time module to measure how long the calculation takes.
start = time.time()

# Calculate similarity 1000 times to get a measurable duration.
# This proves that vector comparison is extremely fast (milliseconds).
for i in range(1000):
    similarity_score = calculate_similarity(text1, text2)

end = time.time()

# Print total time taken for 1000 calculations
print(end - start)

# Print the final similarity score (0.0 to 1.0)
# A score closer to 1.0 means the texts are very similar in meaning.
print(f"Similarity between the texts: {similarity_score:.4f}")

# Calculate similarity 1000 times to get a measurable duration.
# This proves that vector comparison is extremely fast (milliseconds).
for i in range(1000):
    similarity_score = calculate_similarity_vector(text1, text2)

end = time.time()

# Print total time taken for 1000 calculations
print(end - start)

# Print the final similarity score (0.0 to 1.0)
# A score closer to 1.0 means the texts are very similar in meaning.
print(f"Similarity between the texts: {similarity_score:.4f}")