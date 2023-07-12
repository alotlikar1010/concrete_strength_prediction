import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join("artifacts/data_ingestion",'train.csv')
    test_data_path = os.path.join("artifacts/data_ingestion",'test.csv')
    raw_data_path = os.path.join("artifacts/data_ingestion",'data.csv')


class DataIngestion:

    def __init__(self):
        self.ingestion_Config = DataIngestionConfig()
    

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Started:")

        try:
            logging.info("Data Reading starts here:")
            data = pd.read_csv(os.path.join("notebook/dataset","concrete_data.csv"))
            logging.info("Data Reading Completed:")
            # remove duplicate value
            data.drop_duplicates(inplace=True)

            os.makedirs(os.path.dirname(self.ingestion_Config.raw_data_path),exist_ok=True)
            data.to_csv(self.ingestion_Config.raw_data_path, index=False)
            logging.info("Data splited into train and test")

            train_set ,test_set = train_test_split(data,test_size=.30, random_state=42)

            train_set.to_csv(self.ingestion_Config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_Config.test_data_path, index = False, header = True)

            logging.info("Data Ingestion completed")

            return (
                self.ingestion_Config.train_data_path,
                self.ingestion_Config.test_data_path

            )


        except Exception as e:
            logging.info("Error Occured in data ingestion ")
            raise CustomException(e,sys)
