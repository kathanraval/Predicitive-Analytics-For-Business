import pandas as pd
import numpy as np
import  sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
import  sklearn
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import  pandas as pd
import  seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt




df = pd.read_csv("D:\Kathan\Au Assignment\TOD 310- Predicitive Analytics Business for Business\group_1.csv")
print(df)

df1 = df.copy()

columns_to_standardize = ["Ad Campaign Clicks","Time Spent on Website (Minutes)","Page Load Time (Seconds)","Number of visits"]
scaler = StandardScaler()
df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])
print(df)



# Specify the threshold (95th percentile)
threshold = 0.95

# Identify the 95th percentile value for each of the last 3 columns
percentiles = df.iloc[:, -4:].quantile(threshold)

# Create a boolean mask for rows that are below the threshold for each column
mask = (df.iloc[:, -4:] <= percentiles).all(axis=1)

# Filter the DataFrame to keep only the rows below the threshold
filtered_data = df[mask]

# Reset the index after removing rows
filtered_data.reset_index(drop=True, inplace=True)

# output_file_path = "D:\\Kathan\\Au Assignment\\TOD 310- Predicitive Analytics Business for Business\\standardized_new_1.csv"
# filtered_data.to_csv(output_file_path, index=False)

df = pd.read_csv("D:\Kathan\Au Assignment\TOD 310- Predicitive Analytics Business for Business\standardized_new_1.csv")

df3 = df.copy()
df4 = df.copy()
status = pd.get_dummies(df["device type"],drop_first=True)
df = pd.concat([df,status],axis=1)
df = df.drop(["device type"],axis=1)

print(df.columns)

y = df["Conversion Rate"]
X = df[["Ad Campaign Clicks",	"Time Spent on Website (Minutes)",	"Page Load Time (Seconds)",	"Number of visits","Tablet","Mobile"]]
X = sm.add_constant(X)
model = sm.OLS(y,X).fit()
print(model.summary())


df = df.drop(["Tablet"],axis=1)

y = df["Conversion Rate"]
X = df[["Ad Campaign Clicks",	"Time Spent on Website (Minutes)",	"Page Load Time (Seconds)",	"Number of visits","Mobile"]]
X = sm.add_constant(X)
model = sm.OLS(y,X).fit()
print(model.summary())

df = df.drop(["Mobile"],axis=1)
y = df["Conversion Rate"]
X = df[["Ad Campaign Clicks",	"Time Spent on Website (Minutes)",	"Page Load Time (Seconds)",	"Number of visits"]]
X = sm.add_constant(X)
model = sm.OLS(y,X).fit()
print(model.summary())

df = df.drop(["Ad Campaign Clicks"],axis=1)

y = df["Conversion Rate"]
X = df[[	"Time Spent on Website (Minutes)",	"Page Load Time (Seconds)",	"Number of visits"]]
X = sm.add_constant(X)
model = sm.OLS(y,X).fit()
print(model.summary())


###### Kmeans


features = df3.drop(["Sr.","device type"],axis=1)
target = df3[["device type"]]

le = LabelEncoder()
target["device type"] = le.fit_transform(target["device type"])



inertias = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(features)
    inertias.append(kmeans.inertia_)

fig, ax = plt.subplots()
ax.plot(range(1, 11), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.figure(figsize=(5, 3))
plt.show()


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=2)
kmeans = KMeans(n_clusters=3, n_init=10)
kmeans.fit(features)
fig, ax = plt.subplots(figsize=(5,3))
ax.scatter(df['Time Spent on Website (Minutes)'], df['Number of visits'], c=kmeans.labels_)
plt.show()

y_pred = kmeans.predict(X_test)
accs = accuracy_score(y_test['device type'], y_pred)
print("Accuracy Score:", accs)
cm = confusion_matrix(y_test['device type'], y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['mobile','tablet','computer'])
disp.plot()
plt.show()



#### MLR 2


status = pd.get_dummies(df3["device type"],drop_first=True)
df3 = pd.concat([df3,status],axis=1)
df3 = df3.drop(["device type"],axis=1)



y = df3["Time Spent on Website (Minutes)"]
X = df3[["Conversion Rate","Ad Campaign Clicks","Page Load Time (Seconds)","Number of visits","Tablet","Mobile"]]
X = sm.add_constant(X)
model = sm.OLS(y,X).fit()
print(model.summary())

df3 = df3.drop(["Tablet"],axis=1)

y = df3["Time Spent on Website (Minutes)"]
X = df3[["Conversion Rate","Ad Campaign Clicks","Page Load Time (Seconds)","Number of visits","Mobile"]]
X = sm.add_constant(X)
model = sm.OLS(y,X).fit()
print(model.summary())

#####
df4 = df4.drop(["device type","Sr."],axis=1)


# For correlation Matrix and Heat Map

correlation_matrix = df.corr()
print(correlation_matrix)
plt.figure(figsize=(5,3))
sns.heatmap(correlation_matrix,annot=True,cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()


# For pairplot
sns.pairplot(df4)
plt.show()

# For Histogram
df4.hist(figsize=(12, 8))
plt.show()

# For Box Plot
sns.boxplot(data=df4)
plt.show()


summary_stats = df4.describe()
mode_values = df4.mode() # For mode
median_values = df4.median() # For Median
variance_values = df4.var() # For Variance
kurtosis_values = df4.kurtosis() # Kurtosis
skewness_values = df4.skew() # Skewness
min = df4.min() # For minimum
max = df4.max() # For Maximum
range = max - min # For Range
sum = df4.sum() # For Sum
count = df4.count() # For Count
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


