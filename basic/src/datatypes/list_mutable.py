ingredients=["water","milk","black tea"]
print(f"Ingredients without sugar: {ingredients}")
ingredients.append("sugar")
print(f"Ingredients with sugar: {ingredients}")
ingredients.remove("milk")
print(f"Ingredients without milk: {ingredients}")

spice_options=["ginger","cardamom"]
tea_ingredients=["water","milk"]
tea_ingredients.extend(ingredients)
print(f"Tea Ingredients: {tea_ingredients}")
tea_ingredients.insert(0,"tea leaves")
print(f"Tea Ingredients after tea leaves: {tea_ingredients}")
last_added = ingredients.pop()
print(f"Last added ingredient: {last_added}")
print(f"Ingredients:{ingredients}")
tea_ingredients.reverse()
print(f"Tea Ingredients Reversed:{tea_ingredients}")
tea_ingredients.sort()
print(f"Tea Ingredients Sorted:{tea_ingredients}")
print(f"Max Tea Ingredients: {max(tea_ingredients)}")
sugar_levels=[1,2,3,4,5,6,7,8,9,10]
print(f"Sugar Levels: {sugar_levels}")
print(f"Maximum Sugar Level: {max(sugar_levels)}")
print(f"Minimum Sugar Level: {min(sugar_levels)}")
base_liquid=["water","milk"]
extra_flavour=["ginger","cardamom"]
full_liquid_mix=base_liquid + extra_flavour
print(f"Full Liquid Mix: {full_liquid_mix}")
strong_brew=["black tea","water"] * 3
print(f"Strong Brew: {strong_brew}")
raw_spice_data = bytearray(b"CINNAMON")
print(f"Bytes Data: {raw_spice_data}")
replaced_spice_data=raw_spice_data.replace(b"CINNA",b"CARD")
print(f"Raw Spice Data: {raw_spice_data}")
print(f"Replaced Spice Data: {replaced_spice_data}")