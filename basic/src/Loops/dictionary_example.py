users=[
    {"id":1,"name": "Gaurav", "coupon": "DISCOUNT10","total":100},
    {"id":2,"name": "Rohit", "coupon": "DISCOUNT20","total":200},
    {"id":3,"name": "Ragini", "coupon": "DISCOUNT30","total":300},
    {"id":4,"name": "Priyanka", "coupon": "DISCOUNT40","total":400},
    {"id":5,"name": "Amit", "coupon": "DISCOUNT50","total":150},
]
discounts ={
    "DISCOUNT10":(0.1,0),
    "DISCOUNT20":(0.2,0),
    "DISCOUNT30":(0.3,0),
    "DISCOUNT40":(0,40),
    "DISCOUNT50":(0,50),
}

for user in users:
    percent, fixed=discounts.get(user["coupon"],(0,0))
    discount=user["total"]*percent+fixed
    print(f"User {user['name']} with coupon {user['coupon']} gets a discount of {discount:.2f} on total {user['total']:.2f}")
