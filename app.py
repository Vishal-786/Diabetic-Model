import streamlit as st
import joblib
import numpy as np
import pickle


st.set_page_config('Predict Diabetes Patient')

st.title('🩺🔬 Diabetes Prediction')

with open('Model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('Scaler.pkl', 'rb') as scaler_file:
    sc = pickle.load(scaler_file)

Pregnancies   = st.number_input('Pregnancy Month',min_value=0,max_value=9)
Glucose       = st.number_input('Glucose',min_value=0)
BloodPressure = st.number_input('Blood Pressure',min_value=15,max_value=315)
SkinThickness = st.number_input('Skin Thickness',min_value=0)
Insulin       = st.number_input('Insulin',min_value=0)
BMI           = st.number_input('BMI',min_value=0)
DPF           = st.number_input('Diabetes Pedigree Function',min_value=0.0,max_value = 1.5,step=0.1,help="The DiabetesPedigreeFunction is a continuous value that represents the family history of diabetes. It measures the likelihood of a person developing diabetes based on their genetic history.")
Age           = st.number_input('Age',min_value=0,max_value=125) 



if st.button('Predict'):

    lst = [Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DPF,Age]
    arr = np.array(lst)
    arr = arr.reshape(1, -1)
    arr = sc.transform(arr)
    pred = model.predict(arr)

    value = pred > 0.5

    if(value==1):
        st.warning('🚫 Diabetic')
    else:
        st.success('💪 Not Diabetic')

