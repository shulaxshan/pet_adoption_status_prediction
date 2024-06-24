import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTraining
from src.components.model_trainer import ModelTrainingConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df1=pd.read_csv('notebook/data/pet_adoption_data.csv')
            df = df1.drop(["PetID"],axis=1)
            logging.info('Read the dataset as dataframe')
           
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set, test_test=train_test_split(df, test_size= 0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_test.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Data ingestion completed")

            return (self.ingestion_config.train_data_path,self.ingestion_config.test_data_path)

        except CustomException as e:
            logging.error(e)

if __name__ == "__main__":
    data_ingestion=DataIngestion()
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

    data_transform = DataTransformation()
    train_arr,test_arr,_ = data_transform.initiate_data_transformation(train_data_path, test_data_path)

    model_trainer = ModelTraining()
    print(model_trainer.initiate_model_trainer(train_arr,test_arr))

    



