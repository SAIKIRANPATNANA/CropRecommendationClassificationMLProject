import os 
import sys
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow 
from src.CropRecommendationSystemClassificationMLProject.logger import logging
from src.CropRecommendationSystemClassificationMLProject.exception import CustomException
from src.CropRecommendationSystemClassificationMLProject.components.data_ingestion import DataIngestion
from src.CropRecommendationSystemClassificationMLProject.components.data_validation import DataValidation
from src.CropRecommendationSystemClassificationMLProject.components.data_transformation import DataTransformation
from src.CropRecommendationSystemClassificationMLProject.utils.utils import *
from dataclasses import dataclass
import warnings as warn
warn.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
mlflow.autolog()

@dataclass 
class ModelTrainingConfig:
    model_path = os.path.join('artifacts','model.pkl')
class ModelTraining:
    def __init__(self):
        self.model_training_config = ModelTrainingConfig()
    def initiate_model_training(self,train_array,test_array):
        try: 
            x_train = train_array[:,:-1]
            y_train = train_array[:,-1]
            x_test = test_array[:,:-1]
            y_test = test_array[:,-1]
            models = {
                    'logistic_regression' : LogisticRegression(),
                    'decision_tree' : DecisionTreeClassifier(),
                    'random_forest' : RandomForestClassifier(),
                    'gaussian_nb' : GaussianNB(),
                    }   
            model_report = dict()
            for i in range(len(models)):
                model = list(models.values())[i]
                model.fit(x_train,y_train)
                y_pred = model.predict(x_test)
                accuracy,_,_,_,_ = evaluate_metrics(y_test,y_pred)
                model_report[list(models.keys())[i]] = accuracy
            best_score = 0
            models = list(model_report.keys())
            scores = list(model_report.values())
            best_model_name = None
            for i in range(len(scores)):
                if best_score < scores[i]:
                    best_score = scores[i]
                    best_model_name = models[i]
            model = None
            # if best_model_name == 'logistic_regression':
            #     params = {
            #             'penalty': ['l1', 'l2'],
            #             'C': [0.001, 0.01, 0.1, 1, 10, 100],
            #             'fit_intercept': [True, False],
            #             'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
            #             }
            #     logistic_regression = LogisticRegression()
            #     grid_search = GridSearchCV(estimator=logistic_regression, param_grid=params, cv=5, scoring='accuracy')
            #     grid_search.fit(x_train, y_train)
            #     best_params = grid_search.best_params_
            #     model = LogisticRegression(**best_params)
            #     model.fit(x_train, y_train)
            #     y_pred = model.predict(x_test)
            #     best_score,_,_,_,_  = evaluate_metrics(y_test,y_pred)
            # elif best_model_name == 'decision_tree':
            #     params = {
            #             'criterion': ['gini', 'entropy'],
            #             'splitter': ['best', 'random'],
            #             'max_depth': [None, 5, 10, 15, 20],
            #             'min_samples_split': [2, 5, 10],
            #             'min_samples_leaf': [1, 2, 4],
            #             'max_features': ['auto', 'sqrt', 'log2']
            #             }
            #     desision_tree = DecisionTreeClassifier()
            #     grid_search = GridSearchCV(estimator=desision_tree, param_grid=params, cv=5, scoring='accuracy')
            #     grid_search.fit(x_train, y_train)
            #     best_params = grid_search.best_params_
            #     model = DecisionTreeClassifier(**best_params)
            #     model.fit(x_train, y_train)
            #     y_pred = model.predict(x_test)
            #     best_score,_,_,_,_  = evaluate_metrics(y_test,y_pred)
            if best_model_name == 'random_forest':
                params = {
                        'criterion': ['gini', 'entropy'],
                        'max_depth': [None, 5, 10, 15, 20],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4],
                        'max_features': ['auto', 'sqrt', 'log2']
                        }
                random_forest = RandomForestClassifier()
                grid_search = GridSearchCV(estimator=random_forest, param_grid=params, cv=2, scoring='accuracy')
                grid_search.fit(x_train, y_train)
                best_params = grid_search.best_params_
                model = RandomForestClassifier(**best_params)
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                best_score,_,_,_,_  = evaluate_metrics(y_test,y_pred)
            # elif best_model_name == 'gaussian_nb':
            #     params = {
            #             'var_smoothing': [1e-09]
            #             }
            #     gaussian_nb = GaussianNB()
            #     grid_search = GridSearchCV(estimator=gaussian_nb, param_grid=params, cv=5, scoring='accuracy')
            #     grid_search.fit(x_train, y_train)
            #     best_params = grid_search.best_params_
            #     model = GaussianNB(**best_params)
            #     model.fit(x_train, y_train)
            #     y_pred = model.predict(x_test)
            #     best_score,_,_,_,_  = evaluate_metrics(y_test,y_pred)
            elif model_name == None:
                raise ValueError("Model not found")
            save_object(
                obj = model,
                file_path = self.model_training_config.model_path
            )
            return self.model_training_config.model_path,best_model_name,best_score
        except CustomException as  e:
            raise (e,sys)

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
        print('Model training is successful.')

            
            
                        
                    
                    