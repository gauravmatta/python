import ollama

from chatresponse import response

ollama.generate(model='gemma3',prompt='Why is the sky blue?')
print(response)