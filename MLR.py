import pandas as pd
import numpy as np
import  sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm


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


