import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.svm import SVC
import streamlit as st

df = pd.read_csv("D:\Kathan\Au Assignment\TOD 310- Predicitive Analytics Business for Business\diabetes.csv")

# Visual Python: Data Analysis > Subset
feautres = df.loc[:, ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
print(feautres)

# Visual Python: Data Analysis > Subset
target = df.loc[:, 'Outcome']
print(target)

# Visual Python: Machine Learning > Data Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(feautres, target, test_size=0.2, random_state=4)

# Visual Python: Machine Learning > Classifier
model = SVC(C=15,kernel="rbf",gamma="scale")

# Visual Python: Machine Learning > Fit/Predict
model.fit(X_train, y_train)

# Visual Python: Machine Learning > Fit/Predict
pred = model.predict(X_test)
print(pred)


# Confusion Matrix, Confusion Matrix Display and Accuracy Score

accs = accuracy_score(y_test,y_pred=pred)
print(accs)
cm = confusion_matrix(y_test,y_pred=pred)
print(cm)
disp = ConfusionMatrixDisplay(cm, display_labels=['1','0'])
disp.plot()
plt.show()

# For Converting this into Streamlit
st.write("Accuracy Score:", accs)
st.pyplot(disp.plot().figure_)