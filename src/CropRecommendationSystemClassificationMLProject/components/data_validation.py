import os 
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from src.CropRecommendationSystemClassificationMLProject.logger import logging
from src.CropRecommendationSystemClassificationMLProject.exception import CustomException
from src.CropRecommendationSystemClassificationMLProject.components.data_ingestion import DataIngestion
from src.CropRecommendationSystemClassificationMLProject.utils.utils import *
from dataclasses import dataclass
import warnings as warn
warn.filterwarnings('ignore')
import mlflow 
mlflow.autolog()

class DataValidation:
    def __init__(self):
        pass
    def  initiate_data_validation(self,raw_data_path):
        try: 
            req_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']
            df = pd.read_csv(raw_data_path)
            act_cols = df.columns
            if(len(req_cols)!=len(act_cols)):
                return False
            if(set(req_cols) != set(act_cols)):
                return False
            return True 
        except CustomException as e:
            raise(e,sys)

if __name__ == "__main__":
    obj = DataIngestion()
    raw_data_path,_,_ = obj.initiate_data_ingestion()
    obj = DataValidation()
    validataion_status = obj.initiate_data_validation(raw_data_path)
    if(validataion_status):
        print('Data validation is successful.')
    
    

