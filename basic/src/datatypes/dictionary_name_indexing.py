chai_order = dict(type="Masala Chai",size="Large",sugar=2)
print(f"chai order = {chai_order}")
chai_recipe = {}
chai_recipe["base"]="black tea"
chai_recipe["liquid"]="milk"
print(f"chai recipe = {chai_recipe}")
print(f"chai recipe base = {chai_recipe['base']}")
del chai_recipe["liquid"]
print(f"chai recipe after deleting liquid = {chai_recipe}")
print(f"is sugar in the chai order? {'sugar' in chai_order}")
chai_new_order = dict(type="Ginger Chai",size="Medium",sugar=1)
print(f"Order details (keys): {chai_new_order.keys()}")
print(f"Order details (values): {chai_new_order.values()}")
print(f"Order details (items): {chai_new_order.items()}")
last_order=chai_order.popitem()
print(f"Order details (last item): {last_order}")
print(f"chai order = {chai_order}")
extra_spices={"cardamom":"crushed","ginger":"sliced"}
chai_recipe.update(extra_spices)
print(f"chai recipe after adding extra spices = {chai_recipe}")
chai_size=chai_order["size"]
print(f"chai recipe's chai size = {chai_size}")
customer_note=chai_order.get("customer_note","No Note")
print(f"customer note = {customer_note}")