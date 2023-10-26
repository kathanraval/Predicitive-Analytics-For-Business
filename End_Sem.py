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
import  seaborn as sns


st.set_page_config(layout='wide', page_title='Tab Test')

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(['MLR','K-Means','KNN',"Logistic","PCA","Naive-Bayes","Time Series","Graphs","Summary"])

with tab1:
    st.header("MLR")
    df = pd.read_csv("D:/Kathan/Au Assignment/TOD 310- Predicitive Analytics Business for Business/GDP_Class_Participation.csv")
    print(df)
    st.write(df)


    # Columns to standardize using MinMaxScaler
    columns_to_standardize = ["GDP (Billions USD)","Government Expenditure (Billions USD)",	"Unemployment Rate (%)","Inflation Rate (%)","Interest Rate (%)"]
    scaler = StandardScaler()
    df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])
    st.write("Standardized data")
    st.write(df)
    print(df)

    df1 = df.copy()

    y = df["GDP (Billions USD)"]
    X = df[["Government Expenditure (Billions USD)"	,"Unemployment Rate (%)","Inflation Rate (%)","Interest Rate (%)"]]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    st.write("The Dependent Variable is: GDP(Billions USD)")
    st.write("The Independent Variables are : Government Expenditure (Billions USD), Unemployment Rate (%), Inflation Rate (%), Interest Rate (%) ")
    print(model.summary())
    st.write(model.summary())

with tab2:
    st.header("K-means")

    # Read the Data
    df = pd.read_csv("D:\Kathan\Au Assignment\TOD 310- Predicitive Analytics Business for Business\KNN\iris.csv")
    print(df)

    # Manuevering the Data
    features = df.drop(["Species"], axis=1)
    target = df[["Species"]]

    # COnverting them into Binary
    le = LabelEncoder()
    target["Species"] = le.fit_transform(target["Species"])

    # For Loop for Elbow Method
    inertias = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(features)
        inertias.append(kmeans.inertia_)

    # Elbow method display code
    fig, ax = plt.subplots()
    ax.plot(range(1, 11), inertias, marker='o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.figure(figsize=(5, 3))
    plt.show()
    st.pyplot(fig)

    # Scatter Plot
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.scatter(df['PetalWidthCm'], df['PetalLengthCm'], c=kmeans.labels_)
    plt.show()
    st.pyplot(fig)

    # Train,Test Fit and Predict
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=2)
    kmeans = KMeans(n_clusters=3, n_init=10)
    kmeans.fit(features)
    y_pred = kmeans.predict(X_test)

    # Accuracy Score Print
    accs = accuracy_score(y_test['Species'], y_pred)
    print("Accuracy Score:", accs)
    st.write("Accuracy Score",accs)

    # Confusion Matrix
    cm = confusion_matrix(y_test['Species'], y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Setosa', 'Versicolor', 'Virginica'])
    disp.plot()
    plt.show()
    st.pyplot(disp.plot().figure_)
