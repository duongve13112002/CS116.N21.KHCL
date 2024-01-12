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
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score, accuracy_score, confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from pandas.api.types import is_object_dtype, is_numeric_dtype, is_bool_dtype
from xgboost import XGBClassifier
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Machine Learning Website")
st.header("1. Upload dataset")
uploaded_file = st.file_uploader("Choose a CSV file")
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

    st.header("2. Display dataset")
    st.write(dataframe)
    
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
    y_check = y.to_numpy()
    y_check = set(y_check.flatten())
    if (len(y_check) != 2):
        raise ValueError("Output need only 2 classes , but choosing have "+str(len(y_check)))
    y = y/np.min(y) - 1    

    st.header("4. Choose hyper parameters")
    train_per = st.slider(
        'Select a range of training dataset',
        0, 100, 80)
    st.write('The training dataset is ', train_per,'%')
    st.write('Therefore, the test dataset is ', 100 - train_per,'%')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100 - train_per) * 0.01, random_state=1)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    st.header("6. Compare different models")
    options = st.multiselect(
    'Models will be used to compared:',
    ['XGBoost', 'Logistic Regression', 'SVM', 'Decision Tree'],
    ['XGBoost', 'Logistic Regression', 'SVM', 'Decision Tree'])
    st.write('Models selected:', options)

    st.header("6. Metrics")
    while (True):
        f1score = st.checkbox('F1-score',value=True)    
        accuracy = st.checkbox('Accuracy') 
        logloss = st.checkbox('Log loss')
        if f1score == True or logloss == True or accuracy == True:
            break
    
    st.header("7. Choose K-Fold Cross-validation or not")
    k_fold = st.checkbox('K-Fold Cross-validation')
    if k_fold == True:
        num = st.number_input('Insert the number of fold:')
        st.write('The number is ', num)
        num = int(num)

    if st.button('Run'):   
        df_acc = pd.DataFrame(columns = ['Models', 'F1-score', 'Accuracy', 'Log loss'])
        if k_fold == True:
            folds = KFold(n_splits = num, shuffle = True, random_state = 1)
            for i in options:
                if i == 'Logistic Regression':
                    #st.subheader('Logistic Regression init')
                    lr = LogisticRegression().fit(X_train, y_train)
                    y_pred = lr.predict(X_test)
                    scores = cross_val_score(LogisticRegression(), X_train, y_train, scoring='f1_macro', cv=folds)
                    scores_2 = cross_val_score(LogisticRegression(), X_train, y_train, scoring='accuracy', cv=folds)
                    scores_3 = cross_val_score(LogisticRegression(), X_train, y_train, scoring='neg_log_loss', cv=folds)
                    for a in range(len(scores)):
                        df_acc = df_acc.append({'Models' : 'Logistic Regression', 'F1-score' : scores[a], 'Accuracy' : scores_2[a], 'Log loss' : abs(scores_3[a])}, ignore_index = True)
                    #plot_confusion_matrix(lr, X_test, y_test, display_labels = lr.classes_)
                    #st.pyplot()
                elif i == 'SVM':
                    #st.subheader('SVM')
                    svm = SVC().fit(X_train, y_train)
                    y_pred = svm.predict(X_test)
                    scores = cross_val_score(SVC(), X_train, y_train, scoring='f1_macro', cv=folds)
                    scores_2 = cross_val_score(SVC(), X_train, y_train, scoring='accuracy', cv=folds)
                    scores_3 = cross_val_score(SVC(), X_train, y_train, scoring='neg_log_loss', cv=folds)
                    for a in range(len(scores)):
                        df_acc = df_acc.append({'Models' : 'SVM', 'F1-score' : scores[a], 'Accuracy' : scores_2[a], 'Log loss' : abs(scores_3[a])}, ignore_index = True)
                    #plot_confusion_matrix(svm, X_test, y_test, display_labels = svm.classes_)
                    #st.pyplot()
                elif i == 'Decision Tree':
                    #st.subheader('Decision Tree')
                    dt = DecisionTreeClassifier().fit(X_train, y_train)
                    y_pred = dt.predict(X_test)
                    scores = cross_val_score(DecisionTreeClassifier(), X_train, y_train, scoring='f1_macro', cv=folds)
                    scores_2 = cross_val_score(DecisionTreeClassifier(), X_train, y_train, scoring='accuracy', cv=folds)
                    scores_3 = cross_val_score(DecisionTreeClassifier(), X_train, y_train, scoring='neg_log_loss', cv=folds)
                    for a in range(len(scores)):
                        df_acc = df_acc.append({'Models' : 'Decision Tree', 'F1-score' : scores[a], 'Accuracy' : scores_2[a], 'Log loss' : abs(scores_3[a])}, ignore_index = True)
                    #plot_confusion_matrix(dt, X_test, y_test, display_labels = dt.classes_)
                    #st.pyplot()
                elif i == 'XGBoost':
                    #st.subheader('XGBoost')
                    xg = XGBClassifier().fit(X_train, y_train)
                    y_pred = xg.predict(X_test)
                    scores = cross_val_score(xg, X_train, y_train, scoring='f1_macro', cv=folds)
                    scores_2 = cross_val_score(xg, X_train, y_train, scoring='accuracy', cv=folds)
                    scores_3 = cross_val_score(xg, X_train, y_train, scoring='neg_log_loss', cv=folds)
                    for a in range(len(scores)):
                        df_acc = df_acc.append({'Models' : 'XGBoost', 'F1-score' : scores[a], 'Accuracy' : scores_2[a], 'Log loss' : abs(scores_3[a])}, ignore_index = True)
                    #plot_confusion_matrix(xg, X_test, y_test, display_labels = xg.classes_)
                    #st.pyplot()
        else:
            for i in options:
                if i == 'Logistic Regression':
                    st.subheader('Logistic Regression init')
                    lr = LogisticRegression().fit(X_train, y_train)
                    y_pred = lr.predict(X_test)
                    f1_lr = f1_score(y_test, y_pred, average='macro')
                    log_lr = log_loss(y_test, y_pred)
                    acc_lr = accuracy_score(y_test, y_pred)
                    df_acc = df_acc.append({'Models' : 'Logistic Regression', 'F1-score' : f1_lr, 'Accuracy' : acc_lr, 'Log loss' : log_lr}, ignore_index = True)
                    ConfusionMatrixDisplay.from_predictions(y_pred, y_test)
                    st.pyplot()
                elif i == 'SVM':
                    st.subheader('SVM')
                    svm = SVC().fit(X_train, y_train)
                    y_pred = svm.predict(X_test)
                    f1_svm = f1_score(y_test, y_pred, average='macro')
                    acc_svm = accuracy_score(y_test, y_pred)
                    log_svm = log_loss(y_test, y_pred)
                    df_acc = df_acc.append({'Models' : 'SVM', 'F1-score' : f1_svm, 'Accuracy' : acc_svm, 'Log loss' : log_svm}, ignore_index = True)
                    ConfusionMatrixDisplay.from_predictions(y_pred, y_test)
                    st.pyplot()
                elif i == 'Decision Tree':
                    st.subheader('Decision Tree')
                    dt = DecisionTreeClassifier().fit(X_train, y_train)
                    y_pred = dt.predict(X_test)
                    f1_dt = f1_score(y_test, y_pred, average='macro')
                    acc_dt = accuracy_score(y_test, y_pred)
                    log_dt = log_loss(y_test, y_pred)
                    df_acc = df_acc.append({'Models' : 'Decision Tree', 'F1-score' : f1_dt, 'Accuracy' : acc_dt, 'Log loss' : log_dt}, ignore_index = True) 
                    ConfusionMatrixDisplay.from_predictions(y_pred, y_test)
                    st.pyplot()
                elif i == 'XGBoost':
                    st.subheader('XGBoost')
                    xg = XGBClassifier().fit(X_train, y_train)
                    y_pred = xg.predict(X_test)
                    f1_xg = f1_score(y_test, y_pred, average='macro')
                    acc_xg = accuracy_score(y_test, y_pred)
                    log_xg = log_loss(y_test, y_pred)
                    df_acc = df_acc.append({'Models' : 'XGBoost', 'F1-score' : f1_xg, 'Accuracy' : acc_xg, 'Log loss' : log_xg}, ignore_index = True)
                    ConfusionMatrixDisplay.from_predictions(y_pred, y_test)
                    st.pyplot()
        if f1score and logloss and accuracy:
            st.write(df_acc)
            st.bar_chart(df_acc['F1-score'])
            st.bar_chart(df_acc['Accuracy'])
            st.bar_chart(df_acc['Log loss'])
        elif f1score and accuracy:
            st.write(df_acc[['Models', 'F1-score', 'Accuracy']])
            st.bar_chart(df_acc['F1-score'])
            st.bar_chart(df_acc['Accuracy'])
        elif f1score and logloss:
            st.write(df_acc[['Models', 'F1-score', 'Log loss']])
            st.bar_chart(df_acc['F1-score'])
            st.bar_chart(df_acc['Log loss'])
        elif f1score:
            st.write(df_acc[['Models', 'F1-score']])
            st.bar_chart(df_acc['F1-score'])
        elif logloss and accuracy:
            st.write(df_acc[['Models', 'Accuracy', 'Log loss']])
            st.bar_chart(df_acc['Accuracy'])
            st.bar_chart(df_acc['Log loss'])
        elif accuracy:
            st.write(df_acc[['Models', 'Accuracy']])
            st.bar_chart(df_acc['Accuracy'])
        else:
            st.write(df_acc[['Models', 'Log loss']])
            st.bar_chart(df_acc['Log loss'])
        
