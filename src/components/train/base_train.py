import os
import pandas as pd
import mlflow
from abc import ABC, abstractmethod
from train_utils import select_first_file, train_and_log_model

class BaseTrain(ABC):
    
    def __init__(self,
                model_name,
                train_data,
                test_data
                ):
        self.model_name = model_name
        self.train_data = train_data
        self.test_data = test_data
        self.clf = None
        self.hyperparams = []

    def load_data(self):
        self.train_df = pd.read_csv(select_first_file(self.train_data))
        self.test_df = pd.read_csv(select_first_file(self.test_data))

    def split_data(self):
        TARGET = 'lbl_exploits_has'

        self.y_train = self.train_df.pop(TARGET)
        self.x_train = self.train_df.values
        self.y_test = self.test_df.pop(TARGET)
        self.x_test = self.test_df.values

    def log(self):
        mlflow.log_metric("num_samples_x_train", self.x_train.shape[0])
        mlflow.log_metric("num_features_x_train", self.x_train.shape[1])

        mlflow.log_metric("num_samples_y_train", self.y_train.shape[0])
        mlflow.log_metric("num_features_y_train", self.y_train.shape[1])

        mlflow.log_metric("num_samples_x_test", self.x_test.shape[0])
        mlflow.log_metric("num_features_x_test", self.x_test.shape[1])

        mlflow.log_metric("num_samples_y_test", self.y_test.shape[0])
        mlflow.log_metric("num_features_y_test", self.y_test.shape[1])
        
        print("hyperparams", self.hyperparams)

        print("x_train", self.x_train)
        print("y_train", self.y_train)
        print("x_test", self.x_test)
        print("y_test", self.y_test)

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    def run(self, *args, **kwargs):
        mlflow.start_run()
        self.load_data()
        self.split_data()
        self.train(*args, **kwargs)
        self.log()
        mlflow.end_run()

    
        