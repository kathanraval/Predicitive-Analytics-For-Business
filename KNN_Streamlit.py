import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Function to preprocess the DataFrame and perform KNN classification
def preprocess_and_knn(df):
    features = df.drop(["Species", "Id"], axis=1)
    target = df[["Species"]]

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=4)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
    accuracy_score_val = accuracy_score(y_true=y_test, y_pred=y_pred)

    return cm, accuracy_score_val

# Streamlit app
st.title("K-Nearest Neighbors (KNN) Classification")

# File upload section
st.write("## Step 1: Upload the CSV file")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Display the original data
    st.write("### Original Data")
    st.write(df.head())

    # Preprocessing and KNN section
    st.write("## Step 2: Preprocessing and KNN Classification")

    cm, accuracy_score_val = preprocess_and_knn(df)

    # Display confusion matrix
    st.write("### Confusion Matrix")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Setosa", "Versicolor", "Verginica"])
    disp.plot()
    plt.show()
    st.pyplot()

    # Display accuracy score
    st.write("### Accuracy Score")
    st.write("Accuracy Score:", accuracy_score_val)
