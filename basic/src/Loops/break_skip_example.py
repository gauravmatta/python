flavours=["Ginger","Out of Flavour","Lemon","Discontinued","Tulsi"]

for flavour in flavours:
    if flavour == "Out of Flavour":
        continue
    if flavour == "Discontinued":
        print("Discontinued item found")
        break
    print(f"Flavour is: {flavour}")
print(f"Out of flavour Loop")