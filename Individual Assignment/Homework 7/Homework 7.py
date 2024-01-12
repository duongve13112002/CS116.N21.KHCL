# Họ và tên: Nguyễn Vũ Dương
# Mã số sinh viên: 20520465

from csv import list_dialects
import streamlit as st
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from pandas.api.types import is_object_dtype, is_numeric_dtype, is_bool_dtype
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
st.title("Machine Learning Website")
st.header("1. Upload dataset")
uploaded_file = st.file_uploader("Choose a CSV file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    df = "./" + uploaded_file.name
    with open(df, "wb") as f:
        f.write(bytes_data) 

    st.header("2. Display dataset")
    dataframe = pd.read_csv(df)
    st.write(dataframe)
    
    st.header("3. Choose input features")
    X = dataframe.iloc[:, :]
    m =[]
    for i in X.columns:
        agree = st.checkbox(i)
        if agree == False:
            X = X.drop(labels=i, axis=1)
        else:
            m.append(i)
    st.write(X)
    flag = 0
    for i in X.columns:
        if is_numeric_dtype(X[i]) == False:
            #ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [i])], remainder='passthrough')
            #X = np.array(ct.fit_transform(X))
            labelencoder = LabelEncoder()
            X[i] = labelencoder.fit_transform(X[i])
            flag = 1
    
    st.header("3.1. Outputs")
    y = dataframe.iloc[:, :]
    y = y.drop(m,axis=1)
    count = 0
    for i in y.columns:
        agree = st.checkbox(i,key = count)
        count +=1
        if agree == False:
            y = y.drop(labels=i, axis=1)
    st.write(y)
    for i in y.columns:
        if is_numeric_dtype(y[i]) == False:
            #ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [i])], remainder='passthrough')
            #X = np.array(ct.fit_transform(X))
            labelencoder = LabelEncoder()
            y[i] = labelencoder.fit_transform(y[i])
            flag = 1
    st.header("4. Choose hyper parameters")
    train_per = st.slider(
        'Select a range of training dataset',
        0, 100, 80)
    st.write('The training dataset is ', train_per,'%')
    st.write('Therefore, the test dataset is ', 100 - train_per,'%')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100 - train_per) * 0.01, random_state=42)
    scalar = StandardScaler()
    scalar.fit(X_train)
    X_train = scalar.transform(X_train)
    X_test = scalar.transform(X_test)
    st.header("5. Metrics")
    while (True):
        MAE = st.checkbox('MAE',value=True)
        MSE = st.checkbox('MSE',value=True)
        if MAE == True or MSE == True:
            break
    st.header("6. Choose one algorithm ")
    while (True):
        LR = st.checkbox('Linear regression',value=True)
        Lgs = st.checkbox('Logistic regression',value=False)
        if MAE == True or MSE == True:
            break
    st.header("7. Choose a number for testing model ")
    test_u =np.array([])
    for i in range (len(m)):
        hi = 'Insert the number of '+str(m[i])+' :'
        num = st.number_input(hi)
        test_u = np.concatenate((test_u, np.array([float(num)])))        
    if st.button('Run'):
        if LR == True:
           st.write('Linear Regression init')
           lr = LinearRegression()
           lr.fit(X_train, y_train)

        elif Lgs == True:
           st.write('Logistic Regression init')
           lr = LogisticRegression()
           lr.fit(X_train, y_train)

        y_pred = lr.predict(X_test)
        hi = scalar.transform(np.reshape(test_u, (1,-1)))
        y_pred_u = lr.predict(hi)
        y_pred_u = y_pred_u.reshape(1,)
        string_hi = 'Predict input by user: '+ str(y_pred_u[0])
        st.write(string_hi)
        
        df_me = pd.DataFrame(columns = ['MAE', 'MSE'])
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        df_me = df_me.append({'MAE' : mae, 'MSE' : mse}, ignore_index = True)
        X_axis = np.arange(1)

        if MAE == True and MSE == True:
            st.write(df_me)
            fig, ax = plt.subplots(figsize=(20, 20))
            plt.bar(X_axis - 0.5/2, df_me['MAE'], width=0.5, color='red', label='MAE')
            plt.bar(X_axis + 0.5/2, df_me['MSE'], width=0.5, color='blue', label='MSE')
            plt.title('Compare MAE and MSE', fontsize=30)
            plt.xlabel('Linear Regression', fontsize=15)
            plt.ylabel('Metrics', fontsize=15)
            plt.yscale('log')
            plt.legend()
            fig.tight_layout()
            plt.grid(True)
            st.pyplot(fig)  
        elif MAE == True:
            st.write(df_me['MAE'])
            st.bar_chart(df_me['MAE'])
        else:
            st.write(df_me['MSE'])
            st.bar_chart(df_me['MSE'])