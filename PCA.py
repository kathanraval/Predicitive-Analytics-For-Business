import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import streamlit as st

df = pd.read_csv("D:\Kathan\Au Assignment\TOD 310- Predicitive Analytics Business for Business\PCA_Data.csv")


# Visual Python: Machine Learning > Dimension
model = PCA(n_components=4, random_state=123)
trans = model.fit_transform(df)
print(trans)
pca_df = pd.DataFrame(data=trans, columns=['PC1','PC2', 'PC3','PC4'])
print(pca_df)
print(pca_df.head(10))


eigenvalues = model.explained_variance_
prop_var = eigenvalues / np.sum(eigenvalues)
plt.figure(figsize=(14,10))
plt.plot(np.arange(1, len(eigenvalues)+1),
         eigenvalues, marker='o')
plt.xlabel('Principal Component',
           size = 20)
plt.ylabel('Eigenvalue',
           size = 20)
plt.title('Figure 2: Scree Plot for Eigenvalues',
          size = 25)
plt.axhline(y=1, color='r',
            linestyle='--')
plt.grid(True)
plt.show()
st.pyplot(plt.show())


model = PCA(n_components=4, random_state=4)
trans = model.fit_transform(df)
pca_df = pd.DataFrame(data=trans, columns=['PC1','PC2', 'PC3','PC4'])
df = pd.DataFrame(model.components_, columns=list(df.columns))



sns.heatmap(df, cmap ='RdYlGn', linewidths = 0.50, annot = True)
plt.show()
st.pyplot(plt.show())