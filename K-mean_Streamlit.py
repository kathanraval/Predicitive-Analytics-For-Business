import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt

# Function to preprocess the DataFrame and perform K-Means clustering
def preprocess_and_kmeans(df):
    features = df.drop(["Species"], axis=1)
    target = df[["Species"]]

    le = LabelEncoder()
    target["Species"] = le.fit_transform(target["Species"])

    # Calculate inertia for Elbow Method plot
    inertias = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(features)
        inertias.append(kmeans.inertia_)

    # Scatter plot for K-Means clustering
    kmeans = KMeans(n_clusters=3, n_init=10)
    kmeans.fit(features)
    scatter_plot = plt.figure(figsize=(5, 3))
    plt.scatter(df['PetalWidthCm'], df['PetalLengthCm'], c=kmeans.labels_)
    plt.title('K-Means Clustering')
    plt.xlabel('Petal Width (cm)')
    plt.ylabel('Petal Length (cm)')

    # Perform K-Means clustering on test data
    y_pred = kmeans.predict(features)
    accuracy_score_val = accuracy_score(target['Species'], y_pred)
    cm = confusion_matrix(target['Species'], y_pred)

    return inertias, scatter_plot, accuracy_score_val, cm

# Streamlit app
st.title("K-Means Clustering Analysis")

# File upload section
st.write("## Step 1: Upload the CSV file")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Display the original data
    st.write("### Original Data")
    st.write(df.head())

    # Preprocessing and K-Means clustering section
    st.write("## Step 2: Preprocessing and K-Means Clustering")

    inertias, scatter_plot, accuracy_score_val, cm = preprocess_and_kmeans(df)

    # Display Elbow Method plot
    st.write("### Elbow Method Plot")
    fig, ax = plt.subplots()
    ax.plot(range(1, 11), inertias, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    st.pyplot(fig)

    # Display scatter plot for K-Means clustering
    st.write("### K-Means Clustering Scatter Plot")
    st.pyplot(scatter_plot)

    # Display accuracy score
    st.write("### Accuracy Score")
    st.write("Accuracy Score:", accuracy_score_val)

    # Display confusion matrix
    st.write("### Confusion Matrix")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Setosa', 'Versicolor', 'Virginica'])
    disp.plot()
    st.pyplot(disp.plot().figure_)
