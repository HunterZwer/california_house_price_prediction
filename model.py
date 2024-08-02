import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
import pandas as pd
import streamlit as st
import numpy as np

california = fetch_california_housing()
X = pd.DataFrame(california.data, columns=california.feature_names)
y = california.target

def sidebar_menu():
    global input_data

    MedInc = st.sidebar.slider("MedInc", min_value=float(X['MedInc'].min()), max_value=float(X['MedInc'].max()),
                               value=float(X['MedInc'].mean()))
    HouseAge = st.sidebar.slider("HouseAge", min_value=float(X['HouseAge'].min()), max_value=float(X['HouseAge'].max()),
                                 value=float(X['HouseAge'].mean()))
    AveRooms = st.sidebar.slider("AveRooms", min_value=float(X['AveRooms'].min()), max_value=float(X['AveRooms'].max()),
                                 value=float(X['AveRooms'].mean()))
    AveBedrms = st.sidebar.slider("AveBedrms", min_value=float(X['AveBedrms'].min()),
                                  max_value=float(X['AveBedrms'].max()), value=float(X['AveBedrms'].mean()))
    Population = st.sidebar.slider("Population", min_value=float(X['Population'].min()),
                                   max_value=float(X['Population'].max()), value=float(X['Population'].mean()))
    AveOccup = st.sidebar.slider("AveOccup", min_value=float(X['AveOccup'].min()), max_value=float(X['AveOccup'].max()),
                                 value=float(X['AveOccup'].mean()))
    Latitude = st.sidebar.slider("Latitude", min_value=float(X['Latitude'].min()), max_value=float(X['Latitude'].max()),
                                 value=float(X['Latitude'].mean()))
    Longitude = st.sidebar.slider("Longitude", min_value=float(X['Longitude'].min()),
                                  max_value=float(X['Longitude'].max()), value=float(X['Longitude'].mean()))

    input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])

def features():
    return input_data


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)
joblib.dump(model, "model.joblib")
