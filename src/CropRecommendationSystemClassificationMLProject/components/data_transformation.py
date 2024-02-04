import os 
import sys
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from src.CropRecommendationSystemClassificationMLProject.logger import logging
from src.CropRecommendationSystemClassificationMLProject.exception import CustomException
from src.CropRecommendationSystemClassificationMLProject.components.data_ingestion import DataIngestion
from src.CropRecommendationSystemClassificationMLProject.components.data_validation import DataValidation
from src.CropRecommendationSystemClassificationMLProject.utils.utils import *
from dataclasses import dataclass
import warnings as warn
warn.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import StandardScaler,LabelEncoder
import mlflow 
mlflow.autolog()

@dataclass  
class DataTransformationConfig:
    preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    def get_data_transformation_preprocessor(self):
        try:
            num_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
            num_pipeline = Pipeline(steps=[
                            ('standard_scaler',StandardScaler())
                            ])
            preprocessor = ColumnTransformer([
                            ('num_pipeline',num_pipeline,num_cols),
                            ])
            return preprocessor
        except CustomException as e:
            return (e,sys)
    def initiate_data_transformation(self,train_data_path,test_data_path):
        try:
            df1 = pd.read_csv(train_data_path)
            df2 = pd.read_csv(test_data_path)
            for df in [df1,df2]:
                df = df.dropna()
                df = df.drop_duplicates()
            df = pd.concat([df1,df2])
            num_df = df.select_dtypes(exclude=object)
            z_scores = ((num_df - num_df.mean()) / num_df.std()).abs()
            outliers_z = (z_scores > 3).any(axis=1)
            outliers_indices_z = outliers_z[outliers_z].index
            Q1 = num_df.quantile(0.15)
            Q3 = num_df.quantile(0.85)
            IQR = Q3 - Q1
            outliers_iqr = ((num_df< (Q1 - 1.5 * IQR)) | (num_df > (Q3 + 1.5 * IQR))).any(axis=1)
            outliers_indices_iqr = outliers_iqr[outliers_iqr].index
            li = []
            for i in outliers_indices_z:
                if(i in outliers_indices_iqr):
                    li.append(i)
            for i in outliers_indices_iqr:
                if(i in outliers_indices_z):
                    li.append(i)
            df = df.drop(li)
            target = 'label'
            train_df,test_df = train_test_split(df,test_size=0.25,random_state=0)
            input_train_df = train_df.iloc[:,:-1]
            target_train_df = train_df.iloc[:,-1]
            input_test_df = test_df.iloc[:,:-1]
            target_test_df = test_df.iloc[:,-1]
            preprocessor = self.get_data_transformation_preprocessor()
            input_train_array = preprocessor.fit_transform(input_train_df)
            input_test_array = preprocessor.transform(input_test_df)
            label_encoder = LabelEncoder()
            target_train_array = label_encoder.fit_transform(target_train_df)
            target_test_array = label_encoder.transform(target_test_df)
            train_array = np.c_[input_train_array,target_train_array]
            test_array = np.c_[input_test_array,target_test_array]
            save_object(
                obj = preprocessor,
                file_path = self.data_transformation_config.preprocessor_path,
            )
            return train_array,test_array
        except CustomException as e:
            raise (e,sys)
            
if __name__ == "__main__":
    obj = DataIngestion()
    raw_data_path,train_data_path,test_data_path = obj.initiate_data_ingestion()
    obj = DataValidation()
    validataion_status = obj.initiate_data_validation(raw_data_path)
    if(not validataion_status):
        print('Data validation is not successful.')
    else:
        obj = DataTransformation()
        train_array,test_array = obj.initiate_data_transformation(train_data_path,test_data_path)
        print('Data transformation is successful.')