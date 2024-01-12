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
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.datasets import load_wine
from pandas.api.types import is_object_dtype, is_numeric_dtype, is_bool_dtype


st.title("Machine Learning Website")
st.header("1. Choose dataset")
while (True):
    available = st.checkbox('Choose Wine dataset',value=True)
    upload = st.checkbox('Upload your own dataset')
    if available == True or upload == True:
        break
 
if available == True:
    wine=load_wine()
    dataframe=pd.DataFrame(data=np.c_[wine['data'],wine['target']],columns=wine['feature_names']+['target'])
    uploaded_file = True
else:
    dataframe =None
    st.header("1.1. Upload dataset")
    uploaded_file = st.file_uploader("Choose a CSV file",)
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        df = "./" + uploaded_file.name
        with open(df, "wb") as f:
            f.write(bytes_data)
        while (True):
            header_file = st.checkbox('Header file available',value=True) 
            none_header_file = st.checkbox('None header file')
            if header_file == True or none_header_file == True:
               break
        if  header_file:             
            dataframe = pd.read_csv(df)
        else:
            dataframe = pd.read_csv(df,header = None)
                    

if dataframe is not None:
    st.header("2. Display dataset")    
    st.write(dataframe)
    
    st.header("3. Choose input features")
    X = dataframe.iloc[:, :]
    m =[]
    for i in X.columns:
        agree = st.checkbox(str(i))
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
        agree = st.checkbox(str(i),key = count)
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100 - train_per) * 0.01, random_state=0)

    st.header("5. PCA")
    text_reduce = 'Reduce from '+str(len(m))+' to 1'
    dim_re = st.radio(
    "Do you want reduce the dimensions?",
    ('Choose your own dimension',text_reduce))
    if dim_re == 'Choose your own dimension':
        n_components = st.number_input('Insert the number of dimensions:')
        st.write('The number of dimensions is ', n_components)
        n_components = int(n_components)
        pca = PCA(n_components=n_components)
        cc = 1
    elif dim_re == text_reduce:
        cc = 2
        pca_lst = []
        for i in range(len(m)):
            pca_lst.append(PCA(n_components=i+1))


    st.header("6. Metrics")
    while (True):
        f1score = st.checkbox('F1-score',value=True)
        accuracy = st.checkbox('Accuracy')
        if f1score == True or accuracy == True:
            break
    
    def calculate(pca_a,X_test,X_train):
        pipeline = make_pipeline(StandardScaler(), LogisticRegression())
        X_train = pca_a.fit_transform(X_train)
        X_test = pca_a.transform(X_test)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_pred_prb = pipeline.predict_proba(X_test)
        f1 = f1_score(y_test, y_pred, average='macro')
        acc = accuracy_score(y_test, y_pred)
        return f1,acc

    def fit_something(pca_a,X_test,X_train):
        df_me = pd.DataFrame(columns = ['F1-score','Accuracy'])
        f1,acc = calculate(pca_a,X_test,X_train)
        df_me = df_me.append({'F1-score' : f1,'Accuracy': acc}, ignore_index = True)
        if f1score and accuracy:
            st.write(df_me)
            st.bar_chart(df_me['F1-score'])
            st.bar_chart(df_me['Accuracy'])
        elif f1score:
            st.write(df_me['F1-score'])
            st.bar_chart(df_me['F1-score'])
        else:
            st.write(df_me['Accuracy'])
            st.bar_chart(df_me['Accuracy'])

    def fit_something_l(pca_a,X_test,X_train):
        f1_lst = []
        f1_lst.append(0)
        acc_lst = []
        acc_lst.append(0)
        df_me = pd.DataFrame(columns = ['F1-score', 'Accuracy'])
        for i in range(len(pca_a)):
            f1,acc = calculate(pca_a[i],X_test,X_train)
            f1_lst.append(f1)
            acc_lst.append(acc)
        df_me['F1-score'] = pd.Series(f1_lst)
        df_me['Accuracy'] = pd.Series(acc_lst)
        if f1score and accuracy:
            st.write(df_me[1:])
            st.bar_chart(df_me['F1-score'][1:])
            st.bar_chart(df_me['Accuracy'][1:])
        elif f1score: 
            st.write(df_me['F1-score'][1:])
            st.bar_chart(df_me['F1-score'][1:])
        else:
            st.write(df_me['Accuracy'][1:])
            st.bar_chart(df_me['Accuracy'][1:])


    if st.button('Run'):
        st.write('PCA init')                
        if cc == 1:
           fit_something(pca,X_test,X_train) 
        if cc == 2:
           fit_something_l(pca_lst,X_test,X_train) 


        