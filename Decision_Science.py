import pandas as pd

# Read the Excel file
excel_file = pd.ExcelFile("decision Science.xlsx")

# Specify the sheet name you want to access
sheet_name = "tourism daily"

# Read the specific sheet
df = pd.read_excel(excel_file, sheet_name=sheet_name)

# Define a function to highlight outliers in red
def highlight_outliers(val):
    color = 'red' if val > 1.96 or val < -1.96 else 'black'
    return f'color: {color}'

# Standardize the columns
df_standardized = (df - df.mean()) / df.std()

# Apply the highlight_outliers function to the entire DataFrame
styled_df = df_standardized.style.applymap(highlight_outliers)

# Display the DataFrame with outliers highlighted
styled_df.to_excel("tourism Daily Outliers.xlsx", engine='openpyxl')

