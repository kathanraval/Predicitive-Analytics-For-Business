import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset
df = pd.read_csv("D:\Kathan\Au Assignment\TOD 310- Predicitive Analytics Business for Business\KNN\iris.csv")
df = df.drop(["Species", "Id"], axis=1)

# Streamlit app
st.title("Visualizations for Iris Dataset")

# Display Correlation Matrix Heatmap
st.write("## Correlation Matrix Heatmap")
correlation_matrix = df.corr()
plt.figure(figsize=(5, 3))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
st.pyplot()

# Display Pairplot
st.write("## Pairplot")
sns.pairplot(df)
st.pyplot()

# Display Histogram
st.write("## Histogram")
df.hist(figsize=(12, 8))
st.pyplot()

# Display Box Plot
st.write("## Box Plot")
plt.figure(figsize=(10, 6))
sns.boxplot(data=df)
plt.title("Box Plot")
st.pyplot()
