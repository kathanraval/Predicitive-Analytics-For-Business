import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import streamlit as st

df = pd.read_csv("C:/Users/LENOVO/Downloads/loan_data.csv")
print(df)
status = pd.get_dummies(df["purpose"], drop_first=True)
df = pd.concat([df, status], axis=1)
df = df.drop(["purpose"], axis=1)
print(df)

# Visual Python: Data Analysis > Subset
features = df.loc[:, ['int.rate','installment','log.annual.inc','dti','fico','days.with.cr.line','revol.bal','revol.util','inq.last.6mths','delinq.2yrs','pub.rec','not.fully.paid','credit_card','debt_consolidation','educational','home_improvement','major_purchase','small_business']]
print(features)

# Visual Python: Data Analysis > Subset
target = df.loc[:, 'credit.policy']
print(target)

# Visual Python: Machine Learning > Data Split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=4)

# Visual Python: Machine Learning > Classifier

model = GaussianNB()

# Visual Python: Machine Learning > Fit/Predict
model.fit(X_train, y_train)

# Visual Python: Machine Learning > Fit/Predict
pred = model.predict(X_test)
print(pred)

# Printing Accuracy Score
accs = accuracy_score(y_test,y_pred=pred)
print(accs)

# Printing Confusion Matrix
cm = confusion_matrix(y_test,y_pred=pred)
print(cm)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()

# For Converting this into Streamlit
st.write("Accuracy Score:", accs)
st.pyplot(disp.plot().figure_)

