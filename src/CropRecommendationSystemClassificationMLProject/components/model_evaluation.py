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
from src.CropRecommendationSystemClassificationMLProject.components.data_transformation import DataTransformation
from src.CropRecommendationSystemClassificationMLProject.components.data_transformation import DataTransformation
from src.CropRecommendationSystemClassificationMLProject.components.model_trainer import ModelTraining
from src.CropRecommendationSystemClassificationMLProject.utils.utils import *
from dataclasses import dataclass
from urllib.parse import urlparse
import mlflow 
mlflow.autolog()
import warnings as warn
warn.filterwarnings('ignore')

@dataclass
class ModelEvalautionConfig:
    confusion_matrix_path = os.path.join('artifacts','confusion_matrix.png')
class ModelEvaluation:
    def __init__(self):
        self.model_evaluation_config = ModelEvalautionConfig()
    def initiate_model_evaluation(self,test_array,model_path):
        try:
            x_test = test_array[:,:-1]
            y_test = test_array[:,-1]
            model = load_object(model_path)
            y_pred = model.predict(x_test)
            accuracy, precision, recall, f1, cm = evaluate_metrics(y_test,y_pred)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            with mlflow.start_run():
                mlflow.log_metric('accuracy',accuracy)
                mlflow.log_metric('precision', precision)
                mlflow.log_metric('recall',recall)
                mlflow.log_metric('f1',f1)
                plt.figure(figsize=(8, 6))
                classes = ['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas',
                            'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
                            'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple',
                            'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee']
                sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=classes, yticklabels=classes)
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title('Confusion Matrix')
                plt.savefig(self.model_evaluation_config.confusion_matrix_path)
                mlflow.log_artifact(self.model_evaluation_config.confusion_matrix_path)
            if tracking_url_type_store != "file":
                        mlflow.sklearn.log_model(model, "model", registered_model_name="ml_model")
            else:
                mlflow.sklearn.log_model(model, "model")
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == '__main__':
    obj = DataIngestion()
    raw_data_path,train_data_path,test_data_path = obj.initiate_data_ingestion()
    obj = DataValidation()
    validataion_status = obj.initiate_data_validation(raw_data_path)
    if(not validataion_status):
        print('Data validation is not successful.')
    else:
        obj = DataTransformation()
        train_array,test_array = obj.initiate_data_transformation(train_data_path,test_data_path)
        obj = ModelTraining()        
        model_path,best_model_name,best_score = obj.initiate_model_training(train_array,test_array)
        print(f'Best model is found to be {best_model_name} with an accuracy score of {best_score}.')
        obj = ModelEvaluation()
        obj.initiate_model_evaluation(test_array,model_path)
        print('Model evaluation is successful.')
        
        
                    
                
                    
        

            