import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import numpy as np
import statsmodels.api as sm

df = pd.read_csv("D:\Kathan\Au Assignment\TOD 310- Predicitive Analytics Business for Business\KNN\iris.csv")
print(df)

features = df.drop(["Species","Id"],axis=1)
target = df[["Species"]]

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
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Setosa",'Versicolor','Verginica'])
disp.plot()
plt.show()

accuracyScore = accuracy_score(y_true=y_test,y_pred=y_pred)
print("accuracy score",accuracyScore)