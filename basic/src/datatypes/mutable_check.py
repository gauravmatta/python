sugar_amount = 2
print(f"Initial sugar amount: {sugar_amount}")
sugar_amount = 12
print(f"Second Initial sugar amount: {sugar_amount}")
print(f"ID of 2: {id(2)}")
print(f"ID of 12: {id(12)}")
print(f"ID of 4: {id(4)}")

spice_mix=set()
print(f"Initial spice mix id {id(spice_mix)}")
spice_mix.add("cumin")
print(f"Spice mix after adding cumin: {spice_mix}")
print(f"Spice mix id after adding cumin: {id(spice_mix)}")
spice_mix.add("paprika")
print(f"Spice mix after adding paprika: {spice_mix}")
print(f"Spice mix id after adding paprika: {id(spice_mix)}")