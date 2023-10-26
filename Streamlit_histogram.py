import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load the data
@st.cache
def load_data():
    df = pd.read_csv("D:\Kathan\Au Assignment\TOD 310- Predicitive Analytics Business for Business\KNN\iris.csv")  # Update the path to your CSV file
    return df

# Display the page title and description
st.title("K-Nearest Neighbors Classifier for Iris Dataset")
st.write("This app demonstrates a K-Nearest Neighbors classifier on the Iris dataset.")

# Load the data
df = load_data()

# Display the loaded data
st.write("Loaded Data:")
st.write(df)

# Prepare the data for training
features = df.drop(["Species", "Id"], axis=1)
target = df["Species"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=4)

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict on the test set
y_pred = knn.predict(X_test)

# Display predictions and actual values
st.write("Predictions:")
st.write(y_pred)
st.write("Actual Values:")
st.write(y_test)

# Compute confusion matrix
cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Setosa", "Versicolor", "Verginica"])

# Display confusion matrix plot
st.write("Confusion Matrix:")
fig, ax = plt.subplots()
disp.plot(ax=ax)
st.pyplot(fig)

# Compute and display accuracy score
accuracy_score = accuracy_score(y_true=y_test, y_pred=y_pred)
st.write("Accuracy Score:", accuracy_score)
