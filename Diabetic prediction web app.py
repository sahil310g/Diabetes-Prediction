#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 13:34:56 2023

@author: sahilgupta
"""

import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))


# creating a function for Prediction
def diabetic_pred(input_data):

    # changing the input data to numpy array
    input_data_np = np.asarray(input_data)

    # reshape the array
    input_data_reshape = input_data_np.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshape)
    print(prediction)

    if prediction[0] == 1:
        return 'Person is diabetic'
    else:
        return 'Person is not diabetic'


def main():

    # giving a title
    st.title('Diabetics Prediction Web App')

    # getting input data from the user
    Pregnancies = st.text_input('Number of pregnancies: ')
    Glucose = st.text_input('Glucose Level: ')
    BloodPressure = st.text_input('Blood pressure value: ')
    SkinThickness = st.text_input('Skin thickness value: ')
    Insulin = st.text_input('Insulin level: ')
    BMI = st.text_input('BMI: ')
    DiabetesPedigreeFunction = st.text_input('Diabetes pedigree function value: ')
    Age = st.text_input('Age: ')

    # code for prediction

    diagnosis = ''

    # creating a button for prediction
    if st.button('Diabetics test result'):
        diagnosis = diabetic_pred([Pregnancies, Glucose, BloodPressure,
                                  SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
    st.success(diagnosis)


if __name__ == '__main__':
    main()
        