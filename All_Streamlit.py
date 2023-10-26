import streamlit as st
import  sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import sklearn
from sklearn.neighbors import KNeighborsClassifier
import statsmodels.api as sm
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt

st.set_page_config(layout='wide', page_title='Tab Test')

tab1, tab2, tab3 = st.tabs(['First','Second','Third'])

with tab1:
    st.write("First")
    df = pd.read_csv("D:\Kathan\Au Assignment\TOD 310- Predicitive Analytics Business for Business\MLR\Housing.csv")
    df1 = df.copy()

    columns_to_convert = ["mainroad","guestroom","basement","hotwaterheating","airconditioning","prefarea"]
    binary_mapping = {
        "yes" : 1,
        "no" :0}
    for columns in columns_to_convert:
        df[columns] = df[columns].map(binary_mapping)

    status = pd.get_dummies(df["furnishingstatus"],drop_first=True)
    df = pd.concat([df,status],axis=1)
    # df = df.drop(["furnishingstatus","furnished"],axis=1)
    print(df)

    columns_to_standardize = ["price","area","bedrooms","bathrooms","stories","parking"]
        # scaler = StandardScaler()
        # df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])
        # print(df)

    scaler = MinMaxScaler()
    df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])
    print(df)

    y = df["price"]
    X = df[["area", "bedrooms", "bathrooms", "stories", "mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "parking", "prefarea", "semi-furnished", "unfurnished"]]
    X = sm.add_constant(X)
    model = sm.OLS(y,X).fit()
    print(model.summary())
    st.write(model.summary())




with tab2:
    st.write("Second")
    df = pd.read_csv("D:\Kathan\Au Assignment\TOD 310- Predicitive Analytics Business for Business\KNN\iris.csv")
    st.write(df)

    features = df.drop(["Species", "Id"], axis=1)
    target = df[["Species"]]

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=4)
    st.write("X_train:", X_train)
    st.write("y_train:", y_train)
    st.write("X_test:", X_test)
    st.write("y_test:", y_test)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    st.write("Predicted labels:", y_pred)
    st.write("Actual labels:", y_test)

    cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Setosa", 'Versicolor', 'Verginica'])
    disp.plot()
    plt.show()
    st.pyplot(disp.plot().figure_)

    accuracyScore = accuracy_score(y_true=y_test, y_pred=y_pred)
    st.write("Accuracy score:", accuracyScore)

with tab3:
    st.write('Third')

    df = pd.read_csv("D:\Kathan\Au Assignment\TOD 310- Predicitive Analytics Business for Business\KNN\iris.csv")
    st.write(df)

    features = df.drop(["Species"], axis=1)
    target = df[["Species"]]

    le = LabelEncoder()
    target["Species"] = le.fit_transform(target["Species"])

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
    st.pyplot(fig)

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=2)
    kmeans = KMeans(n_clusters=3, n_init=10)
    kmeans.fit(features)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.scatter(df['PetalWidthCm'], df['PetalLengthCm'], c=kmeans.labels_)
    plt.title('Clustered Data')
    plt.xlabel('PetalWidthCm')
    plt.ylabel('PetalLengthCm')
    st.pyplot(fig)

    y_pred = kmeans.predict(X_test)
    accs = accuracy_score(y_test['Species'], y_pred)
    st.write("Accuracy Score:", accs)

    cm = confusion_matrix(y_test['Species'], y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Setosa', 'Versicolor', 'Virginica'])
    disp.plot()
    plt.title('Confusion Matrix')
    st.pyplot(disp.plot().figure_)