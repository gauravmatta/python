order_amount=int(input("Please enter the amount you would like to order: "))
print(f"Order amount type:{type(order_amount)}")
print(f"Order amount:{order_amount}")
delivery_amount=0 if order_amount > 300 else 30
print(f"Delivery amount type:{type(delivery_amount)}")
print(f"Delivery amount:{delivery_amount}")
