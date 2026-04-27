def serve_tea():
    tea_type="Masala" # local scope
    print(f"Inside function: {tea_type}")

tea_type="Lemon"
serve_tea()
print(f"Outside function: {tea_type}")

def tea_counter():
    tea_order="Lemon"
    def print_tea_order():
        tea_order="Ginger"
        print(f"Inner: {tea_order}")
    print_tea_order()
    print(f"Outside function: {tea_order}")

tea_order="Tulsi"
tea_counter()
print(f"Global : {tea_order}")