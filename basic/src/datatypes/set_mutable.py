essential_spices = {"cardamom","ginger","cinnamon"}
optional_spices = {"cloves","ginger","black pepper"}
all_spices = essential_spices.union(optional_spices)
all_piped_spices = all_spices | essential_spices # another way to union
print(f"essential spices used: {essential_spices}")
print(f"optional spices used: {optional_spices}")
print(f"all spices used: {all_spices}")
print(f"all_piped_spices used: {all_piped_spices}")
common_spices = essential_spices.intersection(optional_spices)
common_amper_spices = essential_spices & optional_spices # another way to intersection
print(f"common_spices used: {common_spices}")
print(f"common_amper_spices used: {common_amper_spices}")

only_in_essential = essential_spices.difference(optional_spices)
only_in_essential_minus = essential_spices - optional_spices
print(f"only_in_essential used: {only_in_essential}")
print(f"only_in_essential_minus used: {only_in_essential_minus}")

print(f"Is 'cloves' in essential spices? {'cloves' in essential_spices}")
print(f"Is 'cloves' in optional spices? {'cloves' in optional_spices}")
