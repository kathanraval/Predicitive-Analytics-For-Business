import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sklearn
from sklearn.neighbors import KNeighborsClassifier

# Load the data
df = pd.read_csv("D:\Kathan\Au Assignment\TOD 310- Predicitive Analytics Business for Business\group_1.csv")


def standardize_data(df):
    # Standardize selected columns
    columns_to_standardize = ["Ad Campaign Clicks", "Time Spent on Website (Minutes)", "Page Load Time (Seconds)",
                              "Number of visits"]
    scaler = StandardScaler()
    df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])
    return df


def filter_data(df, threshold=0.95):
    # Specify the threshold (95th percentile)
    percentiles = df.iloc[:, -4:].quantile(threshold)
    mask = (df.iloc[:, -4:] <= percentiles).all(axis=1)
    filtered_data = df[mask].reset_index(drop=True)
    return filtered_data


def display_summary_statistics(df):
    y = df["Conversion Rate"]
    X = df[["Ad Campaign Clicks", "Time Spent on Website (Minutes)", "Page Load Time (Seconds)", "Number of visits"]]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model.summary()


def display_clusters(df):
    features = df.drop(["Sr.", "Page Load Time (Seconds)"], axis=1)
    inertias = []
    for i in range(1, 11):
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

    kmeans = KMeans(n_clusters=3, n_init=10)
    kmeans.fit(features)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.scatter(df['Ad Campaign Clicks'], df['Time Spent on Website (Minutes)'], c=kmeans.labels_)
    plt.show()


def display_confusion_matrix(df):
    features = df.drop(["Sr."], axis=1)
    target = df[["device type"]]
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=4)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Computer", 'Tablet', 'Mobile'])
    disp.plot()
    plt.show()
    accuracyScore = accuracy_score(y_true=y_test, y_pred=y_pred)
    return accuracyScore


def main():
    st.title("Analytics Dashboard")

    # Create tabs
    tabs = ["Data Overview", "Summary Statistics", "Cluster Analysis", "Confusion Matrix"]
    selected_tab = st.sidebar.selectbox("Select Tab", tabs)

    if selected_tab == "Data Overview":
        st.write("## Data Overview")
        st.write(df)

    elif selected_tab == "Summary Statistics":
        st.write("## Summary Statistics")
        standardized_df = standardize_data(df.copy())
        st.write(display_summary_statistics(standardized_df))

    elif selected_tab == "Cluster Analysis":
        st.write("## Cluster Analysis")
        filtered_df = filter_data(df.copy())
        display_clusters(filtered_df)

    elif selected_tab == "Confusion Matrix":
        st.write("## Confusion Matrix")
        filtered_df = filter_data(df.copy())
        accuracy = display_confusion_matrix(filtered_df)
        st.write("Accuracy Score:", accuracy)


if __name__ == "__main__":
    main()
    # COnversion rate is selected as target
#Conversion rate is selected as the target variable as the data suggests that the other variables have a significant impact on conversion rate. Number of visits has a positive correlation with the conversion rate, device type helps us understand the consumer, ad campaign clicks is positively correlated to conversion rate, Time spend on website has high correlation with conversion rate and the lower the page load time, the higher the conversion rate. Through conversion rate, we can predict future conversion rate, understand user behavior, reduce website error, optimize website and marketing strategy.