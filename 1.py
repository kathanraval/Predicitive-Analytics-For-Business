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



print(df)

y = df["Conversion Rate"]
X = df[["Ad Campaign Clicks",	"Time Spent on Website (Minutes)",	"Page Load Time (Seconds)",	"Number of visits"]]
X = sm.add_constant(X)
model = sm.OLS(y,X).fit()
print(model.summary())

features = df.drop(["Sr.","Page Load Time (Seconds)"],axis=1)
target = df[["device type"]]

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
ax.scatter(df['Ad Campaign Clicks'], df['Time Spent on Website (Minutes)'], c=kmeans.labels_)
plt.show()

y_pred = kmeans.predict(X_test)
accs = accuracy_score(y_test['device type'], y_pred)
print("Accuracy Score:", accs)
cm = confusion_matrix(y_test['device type'], y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['tablet','computer','mobile'])
disp.plot()
plt.show()



df = pd.read_csv("filtered_data.csv")
print(df)

features = df.drop(["Sr."],axis=1)
target = df[["device type"]]

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=4)
print(X_train)
print(y_train)
print(X_test)
print(y_test)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)
print(y_pred)
print(y_test)

cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Computer",'Tablet','Mobile'])
disp.plot()
plt.show()

accuracyScore = accuracy_score(y_true=y_test,y_pred=y_pred)
print("accuracy score",accuracyScore)