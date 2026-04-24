device_status=input("Please enter your device status: ")
temperature = int(input("Please enter your temperature: "))
if device_status == "active":
    if temperature >35:
        print("Device is active and temperature is above 35 degrees. Please turn on the cooling system.")
    else:
        print("Device is active and temperature is at or below 35 degrees. No action needed.")
elif device_status == "inactive":
    print("Device is inactive and temperature is at or below 35 degrees. No action needed.")
else:
    print("Invalid input. Please enter 'active' or 'inactive' for device status and a valid temperature.")