names=["Gaurav","Hitesh","Prince","Ravi","Becky","Carlos","Deepak"]
orders=["Pizza","Burger","Pasta","Fries","Salad","Soda","Ice Cream"]
bills=[50,70,100,45,33,23,88]
for name, order,bill in zip(names, orders,bills):
    print(f"Order Ready for: {name} with dish {order} with amount {bill}")
