masala_spices=("cardamom","cloves","cinnamon")
(spice1,spice2,spice3)=masala_spices
print(f"Main Masala Spices: {spice1},{spice2},{spice3}")
ginger_ratio,cardamom_ratio,cinnamon_ratio=0.5,0.3,0.2
print(f"Ginger Ratio: {ginger_ratio}, Cardamom Ratio: {cardamom_ratio}, Cinnamon Ratio: {cinnamon_ratio}")
ginger_ratio,cardamom_ratio=cardamom_ratio,ginger_ratio
print(f"Ginger Ratio: {ginger_ratio}, Cardamom Ratio: {cardamom_ratio}, Cinnamon Ratio: {cinnamon_ratio}")

# membership

print(f"Is ginger in masala spices ?{'ginger' in masala_spices}")
print(f"Is cinnamon in masala spices ?{'cinnamon' in masala_spices}")
print(f"Is cinnamon in masala spices ?{'Cinnamon' in masala_spices}")