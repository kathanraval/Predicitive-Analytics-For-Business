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

tab1, tab2, tab3, tab4, tab5 = st.tabs(['First','Second','Third',"Fourth","Fifth"])

with tab1:
   st.write("First")
   df = pd.read_csv("D:\Kathan\Au Assignment\TOD 310- Predicitive Analytics Business for Business\group_1.csv")
   print(df)

   df1 = df.copy()
   columns_to_standardize = ["Ad Campaign Clicks", "Time Spent on Website (Minutes)", "Page Load Time (Seconds)",
                              "Number of visits"]
   scaler = StandardScaler()
   df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])
   print(df)

   # Specify the threshold (95th percentile)
   threshold = 0.95

   # Identify the 95th percentile value for each of the last 3 columns
   percentiles = df.iloc[:, -4:].quantile(threshold)

   # Create a boolean mask for rows that are below the threshold for each column
   mask = (df.iloc[:, -4:] <= percentiles).all(axis=1)

   # Filter the DataFrame to keep only the rows below the threshold
   filtered_data = df[mask]

   # Reset the index after removing rows
   filtered_data.reset_index(drop=True, inplace=True)

   # output_file_path = "D:\\Kathan\\Au Assignment\\TOD 310- Predicitive Analytics Business for Business\\standardized_new_1.csv"
   # filtered_data.to_csv(output_file_path, index=False)

   df = pd.read_csv("D:\Kathan\Au Assignment\TOD 310- Predicitive Analytics Business for Business\standardized_new_1.csv")

   df3 = df.copy()
   df4 = df.copy()
   status = pd.get_dummies(df["device type"], drop_first=True)
   df = pd.concat([df, status], axis=1)
   df = df.drop(["device type"], axis=1)

   print(df.columns)

   y = df["Conversion Rate"]
   X = df[["Ad Campaign Clicks", "Time Spent on Website (Minutes)", "Page Load Time (Seconds)", "Number of visits",
            "Tablet", "Mobile"]]
   X = sm.add_constant(X)
   model = sm.OLS(y, X).fit()
   print(model.summary())
   st.write(model.summary())

   df = df.drop(["Tablet"], axis=1)

   y = df["Conversion Rate"]
   X = df[["Ad Campaign Clicks", "Time Spent on Website (Minutes)", "Page Load Time (Seconds)", "Number of visits",
            "Mobile"]]
   X = sm.add_constant(X)
   model = sm.OLS(y, X).fit()
   print(model.summary())
   st.write(model.summary())

   df = df.drop(["Mobile"], axis=1)
   y = df["Conversion Rate"]
   X = df[["Ad Campaign Clicks", "Time Spent on Website (Minutes)", "Page Load Time (Seconds)", "Number of visits"]]
   X = sm.add_constant(X)
   model = sm.OLS(y, X).fit()
   print(model.summary())
   st.write(model.summary())

   df = df.drop(["Ad Campaign Clicks"], axis=1)

   y = df["Conversion Rate"]
   X = df[["Time Spent on Website (Minutes)", "Page Load Time (Seconds)", "Number of visits"]]
   X = sm.add_constant(X)
   model = sm.OLS(y, X).fit()
   print(model.summary())
   st.write(model.summary())

with tab2:
   st.write("Second")
   features = df3.drop(["Sr.", "device type"], axis=1)
   target = df3[["device type"]]

   le = LabelEncoder()
   target["device type"] = le.fit_transform(target["device type"])

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
   st.pyplot(fig)

   X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=2)
   kmeans = KMeans(n_clusters=3, n_init=10)
   kmeans.fit(features)
   fig, ax = plt.subplots(figsize=(5, 3))
   ax.scatter(df['Time Spent on Website (Minutes)'], df['Number of visits'], c=kmeans.labels_)
   plt.show()
   st.pyplot(fig)

   y_pred = kmeans.predict(X_test)
   accs = accuracy_score(y_test['device type'], y_pred)
   print("Accuracy Score:", accs)
   st.write(accs)
   cm = confusion_matrix(y_test['device type'], y_pred)
   disp = ConfusionMatrixDisplay(cm, display_labels=['mobile', 'tablet', 'computer'])
   disp.plot()
   plt.show()
   st.pyplot(disp.plot().figure_)

with tab3:
   st.write('Third')
   status = pd.get_dummies(df3["device type"], drop_first=True)
   df3 = pd.concat([df3, status], axis=1)
   df3 = df3.drop(["device type"], axis=1)

   y = df3["Time Spent on Website (Minutes)"]
   X = df3[
      ["Conversion Rate", "Ad Campaign Clicks", "Page Load Time (Seconds)", "Number of visits", "Tablet", "Mobile"]]
   X = sm.add_constant(X)
   model = sm.OLS(y, X).fit()
   print(model.summary())
   st.write(model.summary())

   df3 = df3.drop(["Tablet"], axis=1)

   y = df3["Time Spent on Website (Minutes)"]
   X = df3[["Conversion Rate", "Ad Campaign Clicks", "Page Load Time (Seconds)", "Number of visits", "Mobile"]]
   X = sm.add_constant(X)
   model = sm.OLS(y, X).fit()
   print(model.summary())
   st.write(model.summary())

with tab4:
   st.write('Fourth')
   df4 = df4.drop(["Sr."],axis=1)
   # For correlation Matrix and Heat Map

   correlation_matrix = df.corr()
   print(correlation_matrix)
   plt.figure(figsize=(5, 3))
   sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
   plt.title("Correlation Matrix")
   plt.show()
   st.pyplot()

   # For pairplot
   sns.pairplot(df4)
   plt.show()
   st.pyplot()

   # For Histogram
   df4.hist(figsize=(12, 8))
   plt.show()
   st.pyplot()

   # For Box Plot
   sns.boxplot(data=df4)
   plt.show()
   st.pyplot()

with tab5:
   st.write("Fifth")
   summary_stats = df4.describe()
   mode_values = df4.mode()  # For mode
   median_values = df4.median()  # For Median
   variance_values = df4.var()  # For Variance
   kurtosis_values = df4.kurtosis()  # Kurtosis
   skewness_values = df4.skew()  # Skewness
   min = df4.min()  # For minimum
   max = df4.max()  # For Maximum
   sum = df4.sum()  # For Sum
   count = df4.count()  # For Count
   print(summary_stats)
   st.write(summary_stats)
   print(mode_values)
   st.write(mode_values)
   print(median_values.T)
   st.write(median_values)
   print(variance_values)
   st.write(variance_values)
   print(kurtosis_values)
   st.write(kurtosis_values)
   print(skewness_values)
   st.write(skewness_values)
   print(min)
   st.write(min)
   print(max)
   st.write(max)
   print(sum)
   st.write(sum)
   print(count)
   st.write(count)





