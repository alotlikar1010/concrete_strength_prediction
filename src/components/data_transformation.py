import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocess_obj_file_path = os.path.join("artifacts/data_transformation", "preprcessor.pkl")

class DataTransformation:
    try:

        def __init__(self) :
            self.datatransformconfig = DataTransformationConfig()

    
        def get_data_transformation_obj(self):
        
            logging.info("Data Transformation started")

            numerical_features = ['cement', 'blast_furnace_slag', 'fly_ash', 'water', 'superplasticizer',
        'coarse_aggregate', 'fine_aggregate ', 'age']
            
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ("scaler", StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
                    ("num_pipeline", num_pipeline, numerical_features)
                ])

            return preprocessor

    except Exception as e:
            raise CustomException(e, sys)
    
    def remote_outliers_IQR(self, col, df):
        try:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)

            iqr = Q3 - Q1

            upper_limit = Q3 + 1.5 * iqr
            lowwer_limit = Q1 - 1.5 * iqr

            df.loc[(df[col]>upper_limit), col] = upper_limit
            df.loc[(df[col]<lowwer_limit), col] = lowwer_limit

            return df

        except Exception as e:
            logging.info("Outluers handling error")
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            numerical_features = ['cement', 'blast_furnace_slag', 'fly_ash', 'water', 'superplasticizer',
        'coarse_aggregate', 'fine_aggregate ', 'age']
            
           #handling outlier starts here
            
            for col in numerical_features:
                self.remote_outliers_IQR(col = col, df = train_data)
            
            logging.info("outlier capped for train data")

            for col in numerical_features:
                self.remote_outliers_IQR(col = col, df = test_data)

            logging.info("outlier capped for test data")
        
            #handling outlier ends here

            preprocess_obj = self.get_data_transformation_obj()

            target_column = "concrete_compressive_strength"
            drop_columns = [target_column]
            logging.info("Splitting train data into dependent and independent features")
            input_feature_train_data = train_data.drop(drop_columns, axis = 1)
            target_feature_train_data = train_data[target_column]

            logging.info("Splitting test data into dependent and independent features")
            input_feature_test_data = test_data.drop(drop_columns, axis = 1)
            target_feature_test_data = test_data[target_column]


            # Apply transformation on our train data and test data
            input_train_arr = preprocess_obj.fit_transform(input_feature_train_data)
            input_test_arr = preprocess_obj.transform(input_feature_test_data)

            # Apply preprocessor object on our train data and test data
            train_array = np.c_[input_train_arr, np.array(target_feature_train_data)]
            test_array = np.c_[input_test_arr, np.array(target_feature_test_data)]

            save_object(file_path=self.datatransformconfig.preprocess_obj_file_path,
                        obj=preprocess_obj)
            
            return (train_array,
                    test_array,
                    self.datatransformconfig.preprocess_obj_file_path)

        except Exception as e:
            raise CustomException(e, sys)   

