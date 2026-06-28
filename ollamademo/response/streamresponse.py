from ollama import chat
stream = chat(
    model='gemma3',
    messages=[{'role': 'user', 'content': 'Why is the earth round?'}],
    stream=True,
)

for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)