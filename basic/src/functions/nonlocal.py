tea_choice="Iris"
def update_order():
    tea_type="Cardamon"
    def kitchen():
        nonlocal tea_type
        tea_type="Masala"
        print(f"Inside kitchen using local scope: {tea_type}")
        global tea_choice
        tea_choice="Lemon"
        print(f"Inside kitchen using global scope: {tea_choice}")
    kitchen()
    print(f"After kitchen: {tea_type}")
update_order()
print(f"Global kitchen: {tea_choice}")