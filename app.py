import streamlit as st
import pandas as pd 
import numpy as  np 
from PIL import Image
import warnings as warn 
warn.filterwarnings("ignore")
from src.CropRecommendationSystemClassificationMLProject.pipelines.prediction_pipeline import *

st.markdown("# Crop Recommendation System🌱", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center; font-size:20px; text-align:center;'>Developed By <span style='color:green ;font-size:25px;'>Sai Kiran Patnana</span></h3>",unsafe_allow_html=True)
N = st.slider("Nitrogen LeveL: ", 0, 250, 0)
P = st.slider("Phosphorus LeveL: ", 0, 250, 0)
K = st.slider("Pottasium Level: ", 0, 250, 0)
temperature = st.number_input("Temperature: ", min_value=0.0, format="%f")
humidity = st.number_input("Humidity: ", min_value=0.0, format="%f")
ph = st.number_input("PH Value: ",min_value=0.0, format="%f")
rainfall = st.number_input("Rainfall Level: ", min_value=0.0, format="%f")

data = CustomData(N,P,K,temperature,humidity,ph,rainfall)
final_data = data.get_data_as_dataframe()
prediction_pipeline = PredictionPipeline()
pred = prediction_pipeline.predict(final_data) 
labels = ['apple',
        'banana',
        'blackgram',
        'chickpea',
        'coconut',
        'coffee',
        'cotton',
        'grapes',
        'jute',
        'kidneybeans',
        'lentil',
        'maize',
        'mango',
        'mothbeans',
        'mungbean',
        'muskmelon',
        'orange',
        'papaya',
        'pigeonpeas',
        'pomegranate',
        'rice',
        'watermelon']

if st.button("Predict Crop"):
        st.success(f"Predicted Suitable Crop.")
        if pred>int(pred)+0.5:
                pred = int(pred)+1
        else: 
                pred = int(pred)
        image = Image.open(f"dataset/stuff/{pred}.jpeg")
        caption = labels[pred]
        st.image(image, caption=caption, use_column_width=True)
