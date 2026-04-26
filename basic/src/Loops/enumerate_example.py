menu=["Green","Lemon","Spiced","Mint"]
for item in menu:
    print(f"Menu item is {item}")

for idx, item in enumerate(menu):
    print(f"Menu item #{idx} is {item}")

for idx, item in enumerate(menu,1):
    print(f"Menu item #{idx} is {item}")