import os 
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from src.CropRecommendationSystemClassificationMLProject.logger import logging
from src.CropRecommendationSystemClassificationMLProject.exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import warnings as warn
warn.filterwarnings('ignore')
import mlflow 
mlflow.autolog()

@dataclass
class DataIngestionConfig:
    raw_data_path = os.path.join('artifacts','raw_data')
    train_data_path = os.path.join('artifacts','train_data')
    test_data_path = os.path.join('artifacts','test_data')

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
    def initiate_data_ingestion(self):
        try:
            logging.info('Data ingestion started.')
            df = pd.read_csv('dataset/Crop_recommendation.csv')
            os.makedirs(os.path.dirname(os.path.join(self.data_ingestion_config.raw_data_path)),exist_ok=True)        
            df.to_csv(self.data_ingestion_config.raw_data_path,index=False)
            train_df,test_df = train_test_split(df,test_size=0.25,random_state=42)
            train_df.to_csv(self.data_ingestion_config.train_data_path,index=False)
            test_df.to_csv(self.data_ingestion_config.test_data_path,index=False)
            logging.info('Data ingestion is successful.')
            return (self.data_ingestion_config.raw_data_path,self.data_ingestion_config.train_data_path,self.data_ingestion_config.test_data_path)
        except Exception as e:
            raise(e,sys)

if __name__ == '__main__':
    obj = DataIngestion()
    raw_data_path,train_data_path,test_data_path = obj.initiate_data_ingestion()
    print('Data ingestion is successful.')









