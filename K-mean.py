import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt

df = pd.read_csv("D:\Kathan\Au Assignment\TOD 310- Predicitive Analytics Business for Business\KNN\iris.csv")
print(df)



features = df.drop(["Species"],axis=1)
target = df[["Species"]]

le = LabelEncoder()
target["Species"] = le.fit_transform(target["Species"])



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
ax.scatter(df['PetalWidthCm'], df['PetalLengthCm'], c=kmeans.labels_)
plt.show()

y_pred = kmeans.predict(X_test)
accs = accuracy_score(y_test['Species'], y_pred)
print("Accuracy Score:", accs)
cm = confusion_matrix(y_test['Species'], y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['Setosa','Versicolor','Virginica'])
disp.plot()
plt.show()