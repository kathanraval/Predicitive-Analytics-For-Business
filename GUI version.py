import tkinter as tk
from tkinter import messagebox

def run_code():
    total_values = int(entry_values.get())
    sum_values = 0

    for i in range(total_values):
        value = int(entry_value_inputs[i].get())
        sum_values += value

    result_label.config(text="Summation: " + str(sum_values))

def run_again():
    run_code()
    user_choice = messagebox.askquestion("Run Again", "Do you want to run the code again?")
    if user_choice == "yes":
        for entry in entry_value_inputs:
            entry.delete(0, tk.END)
        result_label.config(text="Summation:")
    else:
        root.destroy()

root = tk.Tk()
root.title("Summation Calculator")

label_values = tk.Label(root, text="Enter the number of values:")
label_values.pack()

entry_values = tk.Entry(root)
entry_values.pack()

calculate_button = tk.Button(root, text="Calculate Sum", command=run_code)
calculate_button.pack()

entry_value_inputs = []
for i in range(10):  # Assuming up to 10 values
    entry = tk.Entry(root)
    entry_value_inputs.append(entry)
    entry.pack()

result_label = tk.Label(root, text="Summation:")
result_label.pack()

run_again_button = tk.Button(root, text="Run Again", command=run_again)
run_again_button.pack()

root.mainloop()
