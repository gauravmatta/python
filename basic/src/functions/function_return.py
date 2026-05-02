def make_tea():
    return "Here is your masala tea"
return_value=make_tea()
print(return_value)
def print_tea():
    print("Here is your masala tea")
return_value=print_tea()
print(return_value)

def idle_teavendor():
    pass

print("idle_teavendor returns:", idle_teavendor())

def sold_cups():
    return 120
print("sold_cups returns:", sold_cups())

def tea_status(cups_left):
    if cups_left == 0:
        return "Sold Out"
    return "Tea is ready"
print("tea_status returns on 0:", tea_status(0)," and tea_status returns on 50:", tea_status(50))

def tea_report():
    return 100,20, 10 # sold, remaining

sold, remaining, not_paid = tea_report()
print(sold)
print(remaining)