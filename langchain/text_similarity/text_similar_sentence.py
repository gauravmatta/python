# Title: Find the Most Similar Sentence in the Text
#
# Description:
# This script applies the concept of "Semantic Similarity" to a practical problem:
# "Semantic Search".
#
# Given a long paragraph and a specific query (reference sentence), how do we find
# the specific sentence in the paragraph that best matches the query?
#
# Logic:
# 1. Split the long paragraph into individual sentences.
# 2. Calculate the similarity score between the query and EACH sentence.
# 3. Sort the sentences by their score (highest to lowest).
#
# Installation:
# pip install spacy==3.7.4
# python -m spacy download en_core_web_md
#
# How to run:
# python text_similar_sentence.py

import spacy

# Load the SpaCy model.
# Remember: 'en_core_web_md' is required for vector embeddings.
nlp = spacy.load("en_core_web_md")


# --- Function 1: Split Text ---
# SpaCy is smart enough to understand grammar. It knows where a sentence ends
# (looking at periods, exclamation marks, etc.) better than just 'text.split(".")'.
def split_text_into_sentences(text):
    """Split long text into sentences using SpaCy."""
    # Process the entire block of text
    doc = nlp(text)
    # print("Split into sentences")
    # print(doc)
    # for sent in doc.sents:
    #     print(sent)
    # print("\nSplit into sentences")
    # doc.sents is a generator that yields the detected sentences.
    # We convert it into a simple list of strings.
    return [sent.text for sent in doc.sents]


# --- Function 2: Calculate Similarity ---
# This function compares the query against every sentence in our list.
def calculate_similarity(reference_sentence, sentences):
    """Calculate similarity between a reference sentence and a list of sentences."""
    similarities = []

    # Loop through every sentence in the document
    for sentence in sentences:
        # Compute cosine similarity between the reference (query) and the current sentence.
        # Note: calling nlp() on every sentence inside the loop can be slow for huge texts,
        # but works fine for this example.
        similarity_score = nlp(reference_sentence).similarity(nlp(sentence))

        # Store both the score and the sentence text as a tuple: (0.85, "Sentence text...")
        similarities.append((similarity_score, sentence))

    return similarities


# --- Function 3: Rank Results ---
# Sort the list of tuples based on the score.
def reorder_sentences_by_similarity(similarities):
    """Reorder sentences based on similarity scores."""
    # key=lambda x: x[0]: Sort based on the first item in the tuple (the score).
    # reverse=True: Sort descending (highest score first).
    return sorted(similarities, key=lambda x: x[0], reverse=True)


# --- Example Usage ---

# The "Database" text we want to search through.
long_text = """
Natural language processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language. 
The ultimate goal of NLP is to enable computers to understand, interpret, and generate human language in a valuable way. 
Applications of NLP include language translation, sentiment analysis, and chatbots. 
As technology advances, NLP continues to evolve and improve, making it an exciting area of study.
"""

# The "Query" - we want to find sentences similar to this one.
reference_sentence = "NLP enables computers to understand human language."

# Step 1: Break paragraph into sentences
sentences = split_text_into_sentences(long_text)

# print("Sentences: " + str(sentences))

# Step 2: Score each sentence against the query
similarities = calculate_similarity(reference_sentence, sentences)

# print("Similarities: " + str(similarities))

# Step 3: Rank them
ordered_sentences = reorder_sentences_by_similarity(similarities)
# print("Ordered Sentences: " + str(ordered_sentences))

# Step 4: Display results
print("Sentences ordered by similarity to the reference sentence:")
for score, sentence in ordered_sentences:
    print(f"Similarity: {score:.4f} - Sentence: {sentence}")