import os
import pickle as pkl 
import warnings as warn
warn.filterwarnings('ignore')
from src.CropRecommendationSystemClassificationMLProject.logger import logging
from src.CropRecommendationSystemClassificationMLProject.exception import CustomException
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def save_object(obj,file_path):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path, 'wb') as f:
            pkl.dump(obj,f)
    except CustomException as e:
        raise (e,sys)

def load_object(file_path):
    try: 
        with open(file_path, 'rb') as f:
            obj = pkl.load(f)
            return obj 
    except CustomException as e:
        raise (e,sys)

def evaluate_metrics(y_test, y_pred):
    try: 
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred)
        return accuracy, precision, recall, f1, cm
    except CustomException as e:
        raise (e,sys)


    