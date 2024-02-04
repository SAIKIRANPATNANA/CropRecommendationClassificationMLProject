import os 
import sys
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from src.CropRecommendationSystemClassificationMLProject.logger import logging
from src.CropRecommendationSystemClassificationMLProject.exception import CustomException
from src.CropRecommendationSystemClassificationMLProject.utils.utils import *
from dataclasses import dataclass
import mlflow 
mlflow.autolog()
import warnings as warn
warn.filterwarnings('ignore')

class PredictionPipeline:
    def __init__(self):
        pass 
    def predict(self,data):
        try: 
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts','model.pkl')
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            preprocessed_data = preprocessor.transform(data)
            pred = model.predict(preprocessed_data)
            return pred
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
                 N:str,
                 P:int,
                 K:int,
                 temperature:float,
                 humidity:float,
                 ph:float,
                 rainfall:float,
                 ):
        self.N = N
        self.P = P
        self.K = K,
        self.temperature = temperature
        self.humidity = humidity
        self.ph = ph
        self.rainfall = rainfall

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                                        'N': self.N,
                                        'P': self.P,
                                        'K': self.K,
                                        'temperature': self.temperature,
                                        'humidity': self.humidity,
                                        'ph': self.ph,
                                        'rainfall': self.rainfall
                                    }
            df = pd.DataFrame(custom_data_input_dict)
            return df
        except Exception as e:
            raise customexception(e,sys)