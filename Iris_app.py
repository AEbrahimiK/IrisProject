# streamlit run Iris_app.py
import streamlit as st
import pandas as pd

st.title( 'وب اپلیکیشن پیش‌بینی نوع گل زنبق' )
st.write( '⬅️  برای شروع، از قسمت چپ صفحه پارامترهای ورودی را تنظیم کنید' )
st.sidebar.header( '⬇️پارامترهای ورودی را تنظیم کنید' )
st.sidebar.write( '(مقادیر به سانتیمتر هستند)' )
def user_input_features():
    sepal_length    = st.sidebar.slider( 'طول کاسبرگ',   4.3, 7.9, 5.4 )
    sepal_width     = st.sidebar.slider( 'پهنای کاسبرگ', 2.0, 4.4, 3.4 )
    petal_length    = st.sidebar.slider( 'طول گلبرگ',    1.0, 6.9, 1.3 )
    petal_width     = st.sidebar.slider( 'پهنای گلبرگ',  0.1, 2.5, 0.2 )
    data            = { 'SepalLengthCm'  : sepal_length,
                        'SepalWidthCm'   : sepal_width,
                        'PetalLengthCm'  : petal_length,
                        'PetalWidthCm'   : petal_width }
    features = pd.DataFrame(data, index=['➡️'])
    return features

df = user_input_features()

st.subheader( ':پارامترهایی که انتخاب شدند' )
st.write( df )

from joblib import load
iris_final_model = load( 'iris_final_model.joblib' )
# from sklearn.linear_model import LogisticRegression
prediction = iris_final_model.predict(df)
prediction_proba = iris_final_model.predict_proba(df)

iris = pd.read_csv( r"C:\0Station\5.   AI Career\Projects\IRIS\Iris data\Iris.csv" )
st.subheader( ':اسامی و اندیس نوع گل‌های زنبق' )
st.write( iris['Species'].unique() )

st.subheader( ':پیش‌بینی' )
st.write( iris.Species.unique() == prediction )

st.subheader( ' [پیوست: درصدهای پیش‌بینی] ' )
st.write( prediction_proba )

st.title('info:')
st.write('''
#### * Classifier: logistic Regressor
#### * Machine Learning with Iris Project
#### * Amirreza Ebrahimi - Winter 2023
 ''')