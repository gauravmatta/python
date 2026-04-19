class Tea:
    def __init__(self,sweetness,milk_level):
        self.sweetness=sweetness
        self.milk_level=milk_level

    def sip(self):
        print("You take a sip of the tea. It's sweet and creamy.")
    def add_sugar(self,amount):
        self.sweetness+=amount
        print(f"You add {amount} units of sugar. Sweetness is now {self.sweetness}.")
myTea = Tea(sweetness=3,milk_level=4)

myTea.add_sugar(5)
myTea.sip()