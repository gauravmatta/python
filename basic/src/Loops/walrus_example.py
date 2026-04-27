from Loops.break_skip_example import flavour

value=13
remainder=value%5
if remainder:
    print(f"Not divisible by 5,remainder is {remainder}")

#using Walrus
value=13

if (remainder := value%5):
    print(f"Used Walrus Not divisible by 5,remainder is {remainder}")

#Example 1
available_sizes=["small","medium","large"]
if(requested_size := input("Enter your tea cup size: ")) in available_sizes:
    print(f"Serving Requested size {requested_size} tea")
else:
    print(f"Sorry we don't have {requested_size} tea")

#Example 2
flavours=["masala","ginger","lemon","mint"]
print("Available flavours: ",flavours)
while(flavour := input("Enter your tea cup flavours: ")) not in flavours:
    print(f"Sorry we don't have {flavour} tea")

print(f"You choose {flavour} tea")