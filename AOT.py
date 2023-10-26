import matplotlib.pyplot as plt
import  pandas as pd
import  seaborn as sns


df = pd.read_csv("D:\Kathan\Au Assignment\TOD 310- Predicitive Analytics Business for Business\KNN\iris.csv")
df = df.drop(["Species","Id"],axis=1)
summary_stats = df.describe()
mode_values = df.mode() # For mode
median_values = df.median() # For Median
variance_values = df.var() # For Variance
kurtosis_values = df.kurtosis() # Kurtosis
skewness_values = df.skew() # Skewness
min = df.min() # For minimum
max = df.max() # For Maximum
range = max - min # For Range
sum = df.sum() # For Sum
count = df.count() # For Count
print(summary_stats)
print(mode_values)
print(median_values.T)
print(variance_values)
print(kurtosis_values)
print(skewness_values)
print(min)
print(max)
print(range)
print(sum)
print(count)