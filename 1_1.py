import pandas as pd
from sklearn.preprocessing import StandardScaler

# Read the CSV file
df = pd.read_csv("D:\Kathan\Au Assignment\TOD 310- Predicitive Analytics Business for Business\standardized_new.csv")



# Specify the threshold (95th percentile)
threshold = 0.95

# Identify the 95th percentile value for each of the last 3 columns
percentiles = df.iloc[:, -3:].quantile(threshold)

# Create a boolean mask for rows that are below the threshold for each column
mask = (df.iloc[:, -3:] <= percentiles).all(axis=1)

# Filter the DataFrame to keep only the rows below the threshold
filtered_data = df[mask]

# Reset the index after removing rows
filtered_data.reset_index(drop=True, inplace=True)




# output_file_path = "D:\\Kathan\\Au Assignment\\TOD 310- Predicitive Analytics Business for Business\\standardized_new_1.csv"
# filtered_data.to_csv(output_file_path, index=False)
