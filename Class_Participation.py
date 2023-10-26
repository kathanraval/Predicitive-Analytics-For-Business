import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import streamlit as st
import matplotlib.pyplot as plt
import  seaborn as sns

st.set_page_config(layout='wide', page_title='Tab Test')

tab1, tab2, tab3, tab4, tab5 = st.tabs(['GDP MLR','Production & Manufacturing MLR','Advertisement MLR',"Test MLR","Coffe MLR"])

with tab1:
    st.header("Anushka : anushka.l@ahduni.edu.in, Kathan: kathan.r@ahduni.edu.in, Mayank: mayank.m1@ahduni.edu.in")
    st.write("GDP MLR")
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

    st.write("Dropping Unemployment Rate (%)")

    df = df.drop(["Unemployment Rate (%)"], axis=1)

    y = df["GDP (Billions USD)"]
    X = df[["Government Expenditure (Billions USD)"	,"Inflation Rate (%)","Interest Rate (%)"]]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    print(model.summary())
    st.write("The Dependent Variable is: GDP(Billions USD)")
    st.write("Government Expenditure (Billions USD), Inflation Rate (%), Interest Rate (%) ")
    st.write(model.summary())

    st.write("Descriptive Stats")

    summary_stats = df.describe()
    mode_values = df.mode()  # For mode
    median_values = df.median()  # For Median
    variance_values = df.var()  # For Variance
    kurtosis_values = df.kurtosis()  # Kurtosis
    skewness_values = df.skew()  # Skewness
    min = df.min()  # For minimum
    max = df.max()  # For Maximum
    sum = df.sum()  # For Sum
    count = df.count()  # For Count
    st.write(summary_stats)
    st.write(mode_values)
    st.write(median_values)
    st.write(variance_values)
    st.write(kurtosis_values)
    st.write(skewness_values)


   # For correlation Matrix and Heat Map
    df1 =df1.drop(["GDP (Billions USD)"],axis=1)
    correlation_matrix = df1.corr()
    print(correlation_matrix)
    plt.figure(figsize=(5, 3))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()
    st.pyplot()

    # For pairplot
    sns.pairplot(df1)
    plt.show()
    st.pyplot()

    # For Histogram
    df1.hist(figsize=(12, 8))
    plt.show()
    st.pyplot()

    # For Box Plot
    sns.boxplot(data=df1)
    plt.show()
    st.pyplot()


with tab2:
    st.write("Production & Manufacturing MLR")
    df = pd.read_csv("D:/Kathan/Au Assignment/TOD 310- Predicitive Analytics Business for Business/Production&Manufacturing.csv")
    print(df)
    st.write(df)

    # Columns to standardize using MinMaxScaler
    columns_to_standardize = ["Production Output (Units)","Manufacturing Labor Hours","Quality Control Failures (Count)","Raw Material Costs (Thousands USD)","Machine Maintenance Costs (Thousands USD)","Energy Consumption (KWh)"]
    scaler = StandardScaler()
    df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])
    print(df)
    st.write("Standardized data")
    st.write(df)

    df1 = df.copy()

    y = df["Production Output (Units)"]
    X = df[["Manufacturing Labor Hours","Quality Control Failures (Count)","Raw Material Costs (Thousands USD)","Machine Maintenance Costs (Thousands USD)","Energy Consumption (KWh)"]]
    X = sm.add_constant(X)
    st.write("The Dependent Variable is: Production Output (units)")
    st.write("The Independent Variables are : Manufacturing Labor Hours, Quality Control Failures (Count) , Raw Material Costs (Thousands USD) , Machine Maintenance Costs (Thousands USD) , Energy Consumption (KWh) ")
    model = sm.OLS(y, X).fit()
    print(model.summary())
    st.write(model.summary())
    st.write("Dropping Machine Maintenance Costs (Thousands USD) ")

    df = df.drop(["Machine Maintenance Costs (Thousands USD)"], axis=1)

    y = df["Production Output (Units)"]
    X = df[["Manufacturing Labor Hours","Quality Control Failures (Count)","Raw Material Costs (Thousands USD)","Energy Consumption (KWh)"]]
    X = sm.add_constant(X)
    st.write("The Dependent Variable is: Production Output (units)")
    st.write( "The Independent Variables are : Manufacturing Labor Hours, Quality Control Failures (Count) , Raw Material Costs (Thousands USD), Energy Consumption (KWh) ")
    model = sm.OLS(y, X).fit()
    print(model.summary())
    st.write(model.summary())

    st.write("Descriptive Stats")

    summary_stats = df.describe()
    mode_values = df.mode()  # For mode
    median_values = df.median()  # For Median
    variance_values = df.var()  # For Variance
    kurtosis_values = df.kurtosis()  # Kurtosis
    skewness_values = df.skew()  # Skewness
    min = df.min()  # For minimum
    max = df.max()  # For Maximum
    sum = df.sum()  # For Sum
    count = df.count()  # For Count
    st.write(summary_stats)
    st.write(mode_values)
    st.write(median_values)
    st.write(variance_values)
    st.write(kurtosis_values)
    st.write(skewness_values)

    # For correlation Matrix and Heat Map
    df1 = df1.drop(["Production Output (Units)"], axis=1)
    correlation_matrix = df1.corr()
    print(correlation_matrix)
    plt.figure(figsize=(5, 3))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()
    st.pyplot()

    # For pairplot
    sns.pairplot(df1)
    plt.show()
    st.pyplot()

    # For Histogram
    df1.hist(figsize=(12, 8))
    plt.show()
    st.pyplot()

    # For Box Plot
    sns.boxplot(data=df1)
    plt.show()
    st.pyplot()

with tab3:
    st.write("Advertisement MLR")
    df = pd.read_csv("D:/Kathan/Au Assignment/TOD 310- Predicitive Analytics Business for Business/ads_1.csv")
    print(df)

    # Columns to standardize using MinMaxScaler
    columns_to_standardize = ["Ad Effectiveness (0-100)",	"Ad Impressions (Thousands)",	"Target Audience Reach (%)",	"Click-Through Rate (%)",	"Conversion Rate (%)",	"Ad Campaign Cost (Thousands USD)",	"Advertising Expense (Thousands USD)",	"Advertising Budget (Thousands USD)",	"Competitor Advertising Spend (Thousands USD)"]
    scaler = StandardScaler()
    df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])
    print(df)
    st.write("Standardized data")
    st.write(df)
    print(df.columns)

    df1 = df.copy()

    y = df["Ad Effectiveness (0-100)"]
    X = df[["Ad Impressions (Thousands)",	"Target Audience Reach (%)",	"Click-Through Rate (%)",	"Conversion Rate (%)",	"Ad Campaign Cost (Thousands USD)",	"Advertising Expense (Thousands USD)",	"Advertising Budget (Thousands USD)",	"Competitor Advertising Spend (Thousands USD)"]]
    X = sm.add_constant(X)
    st.write("The Dependent Variable is: Ad Effectiveness (0-100) ")
    st.write("The Independent Variables are : Ad Impressions (Thousands) , Target Audience Reach (%) , Click-Through Rate (%) , Conversion Rate (%) , Ad Campaign Cost (Thousands USD), Advertising Expense (Thousands USD), Advertising Budget (Thousands USD), Competitor Advertising Spend (Thousands USD) ")
    model = sm.OLS(y, X).fit()
    print(model.summary())
    st.write(model.summary())


    df = df.drop(["Ad Impressions (Thousands)"], axis=1)

    y = df["Ad Effectiveness (0-100)"]
    X = df[["Target Audience Reach (%)", "Click-Through Rate (%)", "Conversion Rate (%)",
            "Ad Campaign Cost (Thousands USD)", "Advertising Expense (Thousands USD)",
            "Advertising Budget (Thousands USD)", "Competitor Advertising Spend (Thousands USD)"]]
    X = sm.add_constant(X)
    st.write("The Dependent Variable is: Ad Effectiveness (0-100) ")
    st.write("The Independent Variables are :  Target Audience Reach (%) , Click-Through Rate (%) , Conversion Rate (%) , Ad Campaign Cost (Thousands USD), Advertising Expense (Thousands USD), Advertising Budget (Thousands USD), Competitor Advertising Spend (Thousands USD) ")
    model = sm.OLS(y, X).fit()
    print(model.summary())
    st.write(model.summary())

    st.write("Descriptive Stats")

    summary_stats = df.describe()
    mode_values = df.mode()  # For mode
    median_values = df.median()  # For Median
    variance_values = df.var()  # For Variance
    kurtosis_values = df.kurtosis()  # Kurtosis
    skewness_values = df.skew()  # Skewness
    min = df.min()  # For minimum
    max = df.max()  # For Maximum
    sum = df.sum()  # For Sum
    count = df.count()  # For Count
    st.write(summary_stats)
    st.write(mode_values)
    st.write(median_values)
    st.write(variance_values)
    st.write(kurtosis_values)
    st.write(skewness_values)

    # For correlation Matrix and Heat Map
    df1 = df1.drop(["Ad Effectiveness (0-100)"], axis=1)
    correlation_matrix = df1.corr()
    print(correlation_matrix)
    plt.figure(figsize=(5, 3))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()
    st.pyplot()

    # For pairplot
    sns.pairplot(df1)
    plt.show()
    st.pyplot()

    fig, ax = plt.subplots(figsize=(12, 8))
    df1.hist(ax=ax)
    plt.close(fig)

    # For Box Plot
    sns.boxplot(data=df1)
    plt.show()
    st.pyplot()
    st.pyplot(fig)


with tab4:
    st.write("Test MLR")
    df = pd.read_csv("D:/Kathan/Au Assignment/TOD 310- Predicitive Analytics Business for Business/Test_1.csv")
    print(df)

    # Columns to standardize using MinMaxScaler
    columns_to_standardize = ["Study Hours (Weekly)",	"Practice Tests Taken (Count)",	"Tutoring Hours (Monthly)",	"Sleep Hours (Night Before Exam)"]
    scaler = StandardScaler()
    df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])
    st.write("Standardized data")
    df1 =df.copy()
    st.write(df)
    print(df)

    y = df["Exam Score (0-100)"]
    X = df[["Study Hours (Weekly)",	"Practice Tests Taken (Count)",	"Tutoring Hours (Monthly)",	"Sleep Hours (Night Before Exam)"]]
    X = sm.add_constant(X)
    st.write("The Dependent Variable is: Exam Score (0-100) ")
    st.write("The Independent Variables are : Ad Impressions (Thousands) , Target Audience Reach (%) , Click-Through Rate (%) , Conversion Rate (%) , Ad Campaign Cost (Thousands USD), Advertising Expense (Thousands USD), Advertising Budget (Thousands USD), Competitor Advertising Spend (Thousands USD) ")
    model = sm.OLS(y, X).fit()
    print(model.summary())
    st.write(model.summary())

    st.write("Practice Tests Taken (Count)")
    df =df.drop(["Practice Tests Taken (Count)"],axis=1)

    y = df["Exam Score (0-100)"]
    X = df[["Study Hours (Weekly)","Tutoring Hours (Monthly)",
            "Sleep Hours (Night Before Exam)"]]
    X = sm.add_constant(X)
    st.write("The Dependent Variable is: Exam Score (0-100) ")
    st.write("Study Hours (Weekly),	Practice Tests Taken (Count),Tutoring Hours (Monthly),Sleep Hours (Night Before Exam)")
    model = sm.OLS(y, X).fit()
    print(model.summary())
    st.write(model.summary())

    st.write("Descriptive Stats")

    summary_stats = df.describe()
    mode_values = df.mode()  # For mode
    median_values = df.median()  # For Median
    variance_values = df.var()  # For Variance
    kurtosis_values = df.kurtosis()  # Kurtosis
    skewness_values = df.skew()  # Skewness
    min = df.min()  # For minimum
    max = df.max()  # For Maximum
    sum = df.sum()  # For Sum
    count = df.count()  # For Count
    st.write(summary_stats)
    st.write(mode_values)
    st.write(median_values)
    st.write(variance_values)
    st.write(kurtosis_values)
    st.write(skewness_values)

    #For correlation Matrix and Heat Map
    df1 = df1.drop(["Exam Score (0-100)"], axis=1)
    correlation_matrix = df1.corr()
    print(correlation_matrix)
    plt.figure(figsize=(5, 3))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()
    st.pyplot()

    # For pairplot
    sns.pairplot(df1)
    plt.show()
    st.pyplot()

    # For Histogram
    df1.hist(figsize=(12, 8))
    plt.show()
    st.pyplot()

    # For Box Plot
    sns.boxplot(data=df1)
    plt.show()
    st.pyplot()

with tab5:
    st.write("Coffee MLR")
    df = pd.read_csv("D:/Kathan/Au Assignment/TOD 310- Predicitive Analytics Business for Business/coffee.csv")
    print(df)
    st.write(df)

    # Columns to standardize using MinMaxScaler
    columns_to_standardize = ["Exam Score", "Study Hours", "Practice Tests Taken", "Tutoring Hours", "Sleep Hours",
                              "Caffeine Consumption "]
    scaler = StandardScaler()
    df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])
    st.write("Standardized data")
    st.write(df)
    print(df)

    df1 = df.copy()

    y = df["Exam Score"]
    X = df[["Study Hours", "Practice Tests Taken", "Tutoring Hours", "Sleep Hours", "Caffeine Consumption "]]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    st.write("The Dependent Variable is: Exam Score")
    st.write(
        "The Independent Variables are : Study Hours, Practice Tests Taken, Tutoring Hours, Sleep Hours, Caffeine Consumption")
    print(model.summary())
    st.write(model.summary())

    st.write("Sleep Hours")

    df = df.drop(["Sleep Hours"], axis=1)

    y = df["Exam Score"]
    X = df[["Study Hours", "Practice Tests Taken", "Tutoring Hours", "Caffeine Consumption "]]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    print(model.summary())
    st.write("The Dependent Variable is: Exam Score")
    st.write("Study Hours", "Practice Tests Taken", "Tutoring Hours", "Caffeine Consumption ")
    st.write(model.summary())

    st.write("Descriptive Stats")

    summary_stats = df.describe()
    mode_values = df.mode()  # For mode
    median_values = df.median()  # For Median
    variance_values = df.var()  # For Variance
    kurtosis_values = df.kurtosis()  # Kurtosis
    skewness_values = df.skew()  # Skewness
    min = df.min()  # For minimum
    max = df.max()  # For Maximum
    sum = df.sum()  # For Sum
    count = df.count()  # For Count
    st.write(summary_stats)
    st.write(mode_values)
    st.write(median_values)
    st.write(variance_values)
    st.write(kurtosis_values)
    st.write(skewness_values)

    # For correlation Matrix and Heat Map
    df1 = df1.drop(["Exam Score"], axis=1)
    correlation_matrix = df1.corr()
    print(correlation_matrix)
    plt.figure(figsize=(5, 3))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()
    st.pyplot()

    # For pairplot
    sns.pairplot(df1)
    plt.show()
    st.pyplot()

    # For Histogram
    df1.hist(figsize=(12, 8))
    plt.show()
    st.pyplot()

    # For Box Plot
    sns.boxplot(data=df1)
    plt.show()
    st.pyplot()




