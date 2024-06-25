import pickle
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)
        
class get_data_for_prediction:
    def __init__(self):
        pass

    def get_data_as_dataframe(self):
        try:
            data1=pd.read_csv('notebook/data/new_data_for_prediction.csv')
            data = data1.drop(["PetID"],axis=1)
            return data
        except Exception as e:
            raise CustomException(e,sys)