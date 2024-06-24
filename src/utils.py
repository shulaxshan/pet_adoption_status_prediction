import os
import sys
import numpy as np 
import pandas as pd
import pickle
from src.exception import CustomException
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score




def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_models(X_train, y_train,X_test,y_test,models):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate classification metrics
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)

            train_f1 = f1_score(y_train, y_train_pred, average='weighted')
            test_f1 = f1_score(y_test, y_test_pred, average='weighted')

            train_precision = precision_score(y_train, y_train_pred, average='weighted')
            test_precision = precision_score(y_test, y_test_pred, average='weighted')

            train_recall = recall_score(y_train, y_train_pred, average='weighted')
            test_recall = recall_score(y_test, y_test_pred, average='weighted')

            # Store the results
            report[list(models.keys())[i]] = {
                'test_accuracy': test_accuracy,
                'test_f1': test_f1,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'train_accuracy': train_accuracy,
                'train_f1': train_f1,
                'train_precision': train_precision,
                'train_recall': train_recall
            }

            return report
        
    except Exception as e:
        raise CustomException(e,sys)