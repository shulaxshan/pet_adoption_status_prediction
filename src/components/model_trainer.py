import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.utils import save_object,evaluate_models
from src.exception import CustomException
from src.logger import logging

@dataclass
class ModelTrainingConfig:trained_model_file_paths = os.path.join("artifacts", 'model.pkl')

class ModelTraining:
    def __init__(self):
        self.model_training_config=ModelTrainingConfig()
        

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models= {
                "AdaBoostClassifier": AdaBoostClassifier(),
                "RandomForestClassifier": RandomForestClassifier(),
                "GradientBoostingClassifier": GradientBoostingClassifier(),
                "KNeighborsClassifier": KNeighborsClassifier()
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]

            # if best_model_score <0.6:
            #     raise CustomException("No best model found")
            # logging.info(f"Best found model on both traing and testing dataset")

            save_object(
                file_path= self.model_training_config.trained_model_file_paths,
                obj = best_model
            )

            predicted = best_model.predict(X_test)
            accuracy_ = accuracy_score(y_test, predicted)

            return best_model, accuracy_
        
        except Exception as e:
            raise CustomException(e,sys)