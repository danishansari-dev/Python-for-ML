# from my_calculator import advance_math
# from my_calculator import basic_math

# from my_calculator.basic_math import add, subtract, multiply, divide
# from my_calculator.advance_math import power, square_root


# after adding __init__.py file -> You can add all the functions of my_calculator
"""__init__.py
    from .basic_math import add, subtract, multiply, divide
    from .advance_math import square_root, power
"""
from my_calculator import add, subtract, divide, power, square_root


print(add(4,5))
print(subtract(4,5))
print(divide(4,5))
