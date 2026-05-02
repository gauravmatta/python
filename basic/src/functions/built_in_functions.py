def tea_flavor(flavor="masala"):
    """ Returns the flavor of tea. Default is 'masala' if no flavor is provided."""
    tea="ginger"
    return flavor

print(tea_flavor.__doc__)
print(tea_flavor.__name__)# Output: masala
help(tea_flavor)

def generate_bill(tea=0,snacks=0):
    """
    Calculate a bill of tea and snacks.
    :param tea: Number of tea cups (10 rupees each)
    :param snacks: Number of snacks (15 rupees each)
    :return: Bill of tea and snacks and thank you message
    """
    total = tea*10 + snacks*15
    return total,"Thank you for giving an opportunity to serve you"