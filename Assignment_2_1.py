import pandas as pd
import streamlit as st
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler

# Read the CSV file
def load_data():
    df = pd.read_csv("Housing.csv")
    return df

df = load_data()

# Sidebar for data preprocessing options
st.sidebar.header('Data Preprocessing')
columns_to_convert = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
binary_mapping = {"yes": 1, "no": 0}
for column in columns_to_convert:
    df[column] = df[column].map(binary_mapping)

status = pd.get_dummies(df["furnishingstatus"])
df = pd.concat([df, status], axis=1)
df = df.drop(["furnishingstatus", "furnished"], axis=1)

columns_to_standardize = ["area", "bedrooms", "bathrooms", "stories", "parking", "price"]
scaler = MinMaxScaler()
df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])

# Display the preprocessed data
st.subheader('Preprocessed Data')
st.write(df.head())

# Multiple Linear Regression
st.header('Multiple Linear Regression')

# Step 1: Fit the initial model
st.subheader('Step 1: Initial Model')
y = df["price"]
X = df[["area", "bedrooms", "bathrooms", "stories", "mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "parking", "prefarea", "semi-furnished", "unfurnished"]]
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
st.text('Initial Model Summary:')
st.text(model.summary())

# Step 2: Remove variables with p-value > 0.05
st.subheader('Step 2: Removing Variables')
st.write("After careful analysis, removing the variable 'bedrooms' and 'semi-furnished' as their p-value is greater than 0.05.")
df = df.drop(["bedrooms", "semi-furnished"], axis=1)

# Step 3: Fit the updated model
st.subheader('Step 3: Updated Model')
y = df["price"]
X = df[["area", "bathrooms", "stories", "mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "parking", "prefarea", "unfurnished"]]
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
st.text('Updated Model Summary:')
st.text(model.summary())
