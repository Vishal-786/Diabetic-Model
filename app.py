import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model

st.set_page_config('Predict Diabetes Patient')

st.title('🩺🔬 Diabetes Prediction')

model = load_model('Model.h5')

with open('Scaler.pkl', 'rb') as f:
    sc = pickle.load(f)


Pregnancies   = st.number_input('Pregnancy Month',min_value=0,max_value=9,step=0.01)
Glucose       = st.number_input('Glucose',min_value=0,step=0.01)
BloodPressure = st.number_input('Blood Pressure',min_value=15,max_value=315,step=0.1,step=0.01)
SkinThickness = st.number_input('Skin Thickness',min_value=0,step=0.01)
Insulin       = st.number_input('Insulin',min_value=0,step=0.01)
BMI           = st.number_input('BMI',min_value=0,step=0.01)
DPF           = st.number_input('Diabetes Pedigree Function',min_value=0.0,max_value = 1.5,step=0.001,help="The DiabetesPedigreeFunction is a continuous value that represents the family history of diabetes. It measures the likelihood of a person developing diabetes based on their genetic history.")
Age           = st.number_input('Age',min_value=0,max_value=125) 



if st.button('Predict'):

    lst = [Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DPF,Age]
    arr = np.array(lst)
    arr = arr.reshape(1, -1)
    arr = sc.transform(arr)
    pred = model.predict(arr)

    value = pred[0] > 0.5

    if(value==1):
        st.warning('🚫 Diabetic')
    else:
        st.success('💪 Not Diabetic')
