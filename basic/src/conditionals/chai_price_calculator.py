tea_size=input("Enter your tea size (small/medium/large): ").lower()
if tea_size == "small":
    print("The price of your tea is 10 Rupees")
elif tea_size == "medium":
    print("The price of your tea is 15 Rupees")
elif tea_size == "large":
    print("The price of your tea is 20 Rupees")
else:
    print("Invalid tea size entered. Please choose small, medium, or large.")
