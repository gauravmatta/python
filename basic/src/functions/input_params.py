tea="Ginger Tea"

def prepare_tea(order):
    print(f"Preparing {order}")

prepare_tea(tea)

teacups=[1,2,3]

def edit_chai(cup):
    print(f"Editing {cup}")
    cup[1]=42
edit_chai(teacups)
print(f"Edited {teacups}")

def make_tea(tea,milk,sugar):
    print(f"Making {tea},{milk},{sugar}")

make_tea("Darjeeling","Yes","Low")
make_tea(tea="Green",sugar="Medium",milk="No")

def special_tea(*ingredients, **extras):
    print("Ingredients", ingredients)
    print("Extras", extras)

special_tea("Tea Leaves","Cardamom","Ginger",milk="Yes",sugar="Low",sweetener="Honey",foam="Yes")

def tea_order(order=[]):
    order.append("Tea Leaves")
    print(f"Order: {order}")

tea_order()
tea_order()

#Better way
def tea_order_default(order=None):
    if order is None:
        order = []
    order.append("Tea Leaves")
    print(f"Default Order: {order}")

tea_order_default()
tea_order_default()