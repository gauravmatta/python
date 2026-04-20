ingredients=["water","milk","black tea"]
ingredients.append("sugar")
print(f"Ingredients: {ingredients}")
ingredients.remove("milk")
print(f"Ingredients: {ingredients}")

spice_options=["ginger","cardamom"]
tea_ingredients=["water","milk","tea leaves"]
tea_ingredients.extend(ingredients)
print(f"Tea Ingredients: {tea_ingredients}")