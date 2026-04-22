import arrow
from collections import namedtuple

print(f"Today's Date: {arrow.now().format('YYYY-MM-DD')}")
print(f"UTC time is : {arrow.utcnow()}")
print(f"UTC time in Rome/Europe is : {arrow.utcnow().to("Europe/Rome")}")
teaProfile=namedtuple("teaProfile",["name","caffeine_mg","flavor_profile"])
green_tea_profile=teaProfile(name="Green Tea",caffeine_mg=30,flavor_profile="Light and Grassy")
print(f"Tea Profile: {teaProfile}")
print(f"Green Tea Profile: {green_tea_profile}")
print(f"Tea Name: {green_tea_profile.name}")
print(f"Caffeine Content: {green_tea_profile.caffeine_mg} mg")
print(f"Flavor Profile: {green_tea_profile.flavor_profile}")