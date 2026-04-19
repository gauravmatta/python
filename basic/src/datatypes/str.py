tea_type="Ginger Tea"
customer_name="Gaurav"
print(f"Order for {customer_name}: {tea_type}")
tea_description="Aromatic and Bold"
print(f"First Word: {tea_description[0:8]}")
print(f"Every 2nd Character of First Word: {tea_description[0:8:2]}")
print(f"Last Word: {tea_description[12:]}")
print(f"Reverse Word: {tea_description[::-1]}")
label_text="Tea Spēcial"
print(f"Original Label: {label_text}")
encoded_label=label_text.encode("utf-8")
print(f"Encoded Label: {encoded_label}")
decoded_label=encoded_label.decode("utf-8")
print(f"Decoded Label: {decoded_label}")
japanese_greeting_text="こんにちは、私の名前はガウラヴです";
print(f"Original Japanese Greeting: {japanese_greeting_text}")
encoded_japanese_greeting=japanese_greeting_text.encode("utf-8")
print(f"Encoded Japanese Greeting: {encoded_japanese_greeting}")
decoded_japanese_greeting=encoded_japanese_greeting.decode("utf-8")
print(f"Decoded Japanese Greeting: {decoded_japanese_greeting}")