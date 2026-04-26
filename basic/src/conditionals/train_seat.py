from unittest import case

seat_type=input("Enter seat type (sleeper/AC/General/Luxury)").lower()
match seat_type:
    case "sleeper":
        print("Sleeper - No AC, beds available!")
        seat_type="sleeper"
    case "AC":
        print("Air Conditioned - comfy ride")
        seat_type="AC"
    case "general":
        print("General - cheapest option - No reservtion")
        seat_type="General"
    case "luxury":
        print("Luxury - Premium experience with all amenities")
        seat_type="Luxury"
    case _:
        print("Invalid seat type entered. Please choose sleeper, AC, General, or Luxury.")
