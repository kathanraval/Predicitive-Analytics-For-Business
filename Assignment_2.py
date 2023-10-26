import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import streamlit as st

# Reading the CSV file
df = pd.read_csv("Housing.csv")

# Converting necessary Variables into binary
columns_to_convert = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
binary_mapping = {
    "yes": 1,
    "no": 0}
for column in columns_to_convert:
    df[column] = df[column].map(binary_mapping)

# Converting Furnishing Status into Dummy variables
status = pd.get_dummies(df["furnishingstatus"])
df = pd.concat([df,status],axis=1)
df = df.drop(["furnishingstatus","furnished"],axis = 1)

# Standardizing the Necessary variables as there is no extremeties
columns_to_standardize = ["area", "bedrooms", "bathrooms", "stories", "parking","price"]
scaler = MinMaxScaler()
df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])

# Running Multiple Linear Regression
y = df["price"]
X = df[["area", "bedrooms", "bathrooms", "stories", "mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "parking", "prefarea", "semi-furnished", "unfurnished"]]
X = sm.add_constant(X)
model = sm.OLS(y,X).fit()
print(model.summary())

# After careful analysis removing the variable "bedrooms" and "semi-furnished" as their p value is greater than 0.05
df = df.drop(["bedrooms","semi-furnished"],axis = 1)

# Running Multiple Linear Regression round 2
y = df["price"]
X = df[["area", "bathrooms", "stories", "mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "parking", "prefarea", "unfurnished"]]
X = sm.add_constant(X)
model = sm.OLS(y,X).fit()
print(model.summary())

# Converting it into Streamlit

# Setting up the Page
st.set_page_config(layout="centered",page_icon="**",page_title="Multiple Linear regression")
st.title("Multiple Linear Regression")

# Displaying the Data
st.write(df.head(100))


