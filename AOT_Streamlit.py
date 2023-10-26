import streamlit as st
import pandas as pd

# Load the Iris dataset
df = pd.read_csv("D:\Kathan\Au Assignment\TOD 310- Predicitive Analytics Business for Business\KNN\iris.csv")
df = df.drop(["Species", "Id"], axis=1)

# Calculate summary statistics
summary_stats = df.describe()
mode_values = df.mode()  # For mode
median_values = df.median()  # For Median
variance_values = df.var()  # For Variance
kurtosis_values = df.kurtosis()  # Kurtosis
skewness_values = df.skew()  # Skewness
min_values = df.min()  # For minimum
max_values = df.max()  # For Maximum
range_values = max_values - min_values  # For Range
sum_values = df.sum()  # For Sum
count_values = df.count()  # For Count

# Streamlit app
st.title("Summary Statistics for Iris Dataset")

# Display summary statistics in a tabular format
st.write("## Summary Statistics")
st.write(summary_stats)

# Display mode values
st.write("## Mode Values")
st.write(mode_values)

# Display median values
st.write("## Median Values")
st.write(median_values.T)

# Display variance values
st.write("## Variance Values")
st.write(variance_values)

# Display kurtosis values
st.write("## Kurtosis Values")
st.write(kurtosis_values)

# Display skewness values
st.write("## Skewness Values")
st.write(skewness_values)

# Display minimum values
st.write("## Minimum Values")
st.write(min_values)

# Display maximum values
st.write("## Maximum Values")
st.write(max_values)

# Display range values
st.write("## Range Values")
st.write(range_values)

# Display sum values
st.write("## Sum Values")
st.write(sum_values)

# Display count values
st.write("## Count Values")
st.write(count_values)
