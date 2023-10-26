import pandas as pd
import numpy as np
from scipy.stats import zscore


# Reading the Excel
df = pd.read_excel("insurance1.csv.xlsx")
print(df)

# Function to remove rows with outlier values using z-scores
def remove_rows_with_outliers(data, z_threshold=1.96):
    z_scores = np.abs(zscore(data))
    outlier_mask = z_scores < z_threshold
    data_without_outliers = data[outlier_mask]
    return data_without_outliers

# Remove rows with outliers from 'Bmi-1', 'Age-1', 'Children-1'
df['Bmi-1'] = remove_rows_with_outliers(df['Bmi-1'])
df['Age-1'] = remove_rows_with_outliers(df['Age-1'])
df['Children-1'] = remove_rows_with_outliers(df['Children-1'])


# Now we have removed rows with outlier values in these columns
print(df)

# Save the modified DataFrame to a new Excel file
# output_file = "insurance1_without_outliers.xlsx"
# df.to_excel(output_file, index=False)

