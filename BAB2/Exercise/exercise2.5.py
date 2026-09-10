# Exercise 2.5
import math

print('Enter a list of float numbers (separated by space): ')
input_string = input()

numbers_text = input_string.split()

for item in numbers_text:
    x = float(item)
    y = math.sin(x)
    print("The sine of " + str(x) + " is " + str(y))