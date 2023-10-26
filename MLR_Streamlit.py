import pandas as pd
import numpy as np
import streamlit as st
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler

# Function to preprocess the DataFrame as per the provided code
def preprocess_data(df):
    # Columns to convert to binary
    columns_to_convert = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
    binary_mapping = {"yes": 1, "no": 0}
    for column in columns_to_convert:
        df[column] = df[column].map(binary_mapping)

    # One-hot encode furnishingstatus
    status = pd.get_dummies(df["furnishingstatus"])
    df = pd.concat([df, status], axis=1)
    df = df.drop(["furnishingstatus", "furnished"], axis=1)

    # Columns to standardize using MinMaxScaler
    columns_to_standardize = ["price", "area", "bedrooms", "bathrooms", "stories", "parking"]
    scaler = MinMaxScaler()
    df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])

    return df

# Streamlit app
st.title("Multiple Linear Regression Analysis")

# File upload section
st.write("## Step 1: Upload the CSV file")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Preprocessing section
    st.write("## Step 2: Data Preprocessing")
    st.write("### Original Data")
    st.write(df.head())

    processed_df = preprocess_data(df)

    st.write("### Processed Data")
    st.write(processed_df.head())

    # Model summary section
    st.write("## Step 3: Model Summary")
    y = processed_df["price"]
    X = processed_df[["area", "bedrooms", "bathrooms", "stories", "mainroad", "guestroom", "basement", "hotwaterheating",
                      "airconditioning", "parking", "prefarea", "semi-furnished", "unfurnished"]]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    st.write("### Regression Model Summary")
    st.write(model.summary())
