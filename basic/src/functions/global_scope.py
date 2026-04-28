tea_type="Plain"

def front_desk():
    def kitchen():
        global tea_type
        tea_type="Irnai"
        print(f"Inside kitchen: {tea_type}")
    kitchen()
    print(f"After kitchen: {tea_type}")
front_desk()
print(f"Global : {tea_type}")