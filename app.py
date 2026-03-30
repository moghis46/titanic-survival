import streamlit as st
import pickle
import numpy as np

model  = pickle.load(open('titanic_model.pkl', 'rb'))
scaler = pickle.load(open('titanic_scaler.pkl', 'rb'))

st.title('🚢 Titanic Survival Prediction')

pclass = st.selectbox('Ticket Class', [1, 2, 3])
sex    = st.selectbox('Sex', ['Male', 'Female'])
age    = st.number_input('Age', min_value=0, max_value=100, value=25)
sibsp  = st.number_input('Siblings/Spouse', min_value=0, max_value=8, value=0)
parch  = st.number_input('Parents/Children', min_value=0, max_value=6, value=0)
fare   = st.number_input('Fare', min_value=0.0, max_value=600.0, value=32.0)
embarked = st.selectbox('Embarked', ['S', 'C', 'Q'])

sex_val      = 1 if sex == 'Female' else 0
embarked_val = {'S': 0, 'C': 1, 'Q': 2}[embarked]

if st.button('Predict'):
    features = np.array([[pclass, sex_val, age, sibsp, parch, fare, embarked_val]])
    scaled   = scaler.transform(features)
    pred     = model.predict(scaled)
    if pred[0] == 1:
        st.success('✅ Survived!')
    else:
        st.error('❌ Not Survived!')