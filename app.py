import streamlit as st
import joblib
from model import features, sidebar_menu
from PIL import Image


model: object = joblib.load("model.joblib")
st.title("California House Price Prediction")
sidebar_menu()
image = Image.open('data/house.jpg')
st.image(image, caption='Прогноз цены')

if st.button("Predict"):
    prediction = model.predict(features())
    st.success(f"Predicted House Price: ${prediction[0] * 100000:.2f}")
    st.error(prediction)
# Run the Streamlit app
