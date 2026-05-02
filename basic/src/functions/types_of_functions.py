def pure_function(cups):
    return cups * 10

print(pure_function(5))

total_tea = 0

# impure function as it modifies global not recommended
def impure_function(cup):
    global total_tea
    total_tea = total_tea + cup

impure_function(5)
print("Total tea after impure function:", total_tea)

def recursive_function(cup):
    print(cup)
    if cup == 0:
        return "All cups poured"
    return recursive_function(cup-1)

print(recursive_function(5))

tea_types=["Lipton","Tetley","Tata Tea","Brooke Bond","Tata Tea"]
strong_tea=list(filter(lambda x: x=="Tata Tea",tea_types))
print(strong_tea)
normal_tea=list(filter(lambda x: x!="Tata Tea",tea_types))
print(normal_tea)