import matplotlib.pyplot as plt
import  pandas as pd
import  seaborn as sns


df = pd.read_csv("D:\Kathan\Au Assignment\TOD 310- Predicitive Analytics Business for Business\KNN\iris.csv")
df = df.drop(["Species","Id"],axis=1)


# For correlation Matrix and Heat Map

correlation_matrix = df.corr()
print(correlation_matrix)
plt.figure(figsize=(5,3))
sns.heatmap(correlation_matrix,annot=True,cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()


# For pairplot
sns.pairplot(df)
plt.show()

# For Histogram
df.hist(figsize=(12, 8))
plt.show()

# For Box Plot
sns.boxplot(data=df)
plt.show()


