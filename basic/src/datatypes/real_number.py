import sys
from fractions import Fraction
from decimal import Decimal

ideal_temp=95.5
current_temp=95.4999999999

print(f"Ideal temperature: {ideal_temp}")
print(f"Current temperature: {current_temp}")
print(f"Difference between ideal and current temperature: {ideal_temp-current_temp}")
print(f"Difference between ideal and current temperature: {Decimal(ideal_temp-current_temp)}")

ideal_temp=95.5
current_temp=95.49

print(f"Ideal temperature: {ideal_temp}")
print(f"Current temperature: {current_temp}")
print(f"Difference between ideal and current temperature: {ideal_temp-current_temp}")
print(sys.float_info)