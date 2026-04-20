is_boiling=True
stir_count = 5
total_actions=stir_count+is_boiling #upcasting boolean to integer (True=1, False=0)
print(f"Total actions: {total_actions}")

milk_present=0
print(f"Is Milk present: {bool(milk_present)}")
milk_present=1
print(f"Is Milk present: {bool(milk_present)}")
milk_present=11
print(f"Is Milk present at 11: {bool(milk_present)}")

water_hot = True
tea_added= False
can_serve = water_hot and tea_added
print(f"Can serve water and tea added: {can_serve}")



if is_boiling:
    print(f"Spice mix boiling: {is_boiling}")
    is_boiling=False