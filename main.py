import pandas as pd
import numpy as np
import matplotlib.pyplot


while True:

    total_values = int(input("Enter the number of values: "))

    sum_values = 0

    for i in range(total_values):
         value = int(input(f"Enter value {i + 1}: "))
         sum_values += value

    print("Summation:", sum_values)

    user_choice = input("Do you want to run the code again? (1 for yes, 2 for no): ")
    if user_choice == "1":
        continue
    elif user_choice == "2":
        break
    else:
        print("Invalid input. Please enter 1 for yes or 2 for no.")




