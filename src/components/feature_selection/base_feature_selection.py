import os
import pandas as pd
import mlflow
from abc import ABC, abstractmethod
from feature_selection_utils import select_first_file, FEATURE_KEY, TARGET

# Indicação do uso do padrão Template Method via nomenclatura
class BaseFeatureSelection(ABC):
    """
    Classe base abstrata para o padrão Template Method.
    Define o esqueleto do algoritmo de seleção de características.
    """

    def __init__(self, train_data : str, 
                 test_data : str, 
                 train_data_feat_sel : str, 
                 test_data_feat_sel : str, 
                 feature_percentage : float, 
                 method_name : str):
        self.train_data = train_data
        self.test_data = test_data
        self.train_data_feat_sel = train_data_feat_sel
        self.test_data_feat_sel = test_data_feat_sel
        self.feature_percentage = feature_percentage
        self.method_name = method_name
    
    def load_data(self):
        self.df_train = pd.read_csv(select_first_file(self.train_data))
        self.df_test = pd.read_csv(select_first_file(self.test_data))

    def split_data(self):
        self.y_train = self.df_train.pop(TARGET)
        self.X_train = self.df_train
        self.y_test = self.df_test.pop(TARGET)
        self.X_test = self.df_test

    @abstractmethod
    def compute_feature_scores(self):
        """
        Método abstrato que deve ser implementado pelas subclasses.
        Responsável por calcular as pontuações das características.
        Infogain: ...
        Gini: ...
        Pearson: ...
        SPEARMAN: ...
        """
        pass

    def select_top_features(self):
        self.feature_scores = self.feature_scores.sort_values(by=self.method_name, ascending=False)
        feature_quantity = int(self.X_train.shape[1] * self.feature_percentage)
        self.top_features = self.feature_scores.head(feature_quantity)[FEATURE_KEY]

    def save_selected_features(self):
        df_train_selected = self.df_train[self.top_features]
        df_test_selected = self.df_test[self.top_features]
        df_train_selected = pd.concat([df_train_selected, self.y_train], axis=1)
        df_test_selected = pd.concat([df_test_selected, self.y_test], axis=1)

        self.df_train = df_train_selected
        self.df_test = df_test_selected

        df_train_selected.to_csv(os.path.join(self.train_data_feat_sel, "feat_sel_data.csv"), index=False)
        df_test_selected.to_csv(os.path.join(self.test_data_feat_sel, "feat_sel_data.csv"), index=False)

    def log_metrics_before_selection(self):
        mlflow.log_metric("num_samples_train_original", self.df_train.shape[0])
        mlflow.log_metric("num_features_train_original", self.df_train.shape[1])
        mlflow.log_metric("num_samples_test_original", self.df_test.shape[0])
        mlflow.log_metric("num_features_test_original", self.df_test.shape[1])
        mlflow.log_metric("feature_percentage", self.feature_percentage)
    
    def log_metrics_after_selection(self):
        mlflow.log_metric("num_features_train_feat_sel", self.df_train.shape[1] - 1)
        mlflow.log_metric("num_features_test_feat_sel", self.df_test.shape[1] - 1)
        print("top_features", self.top_features)

    def run(self):
        """
        Método Template que define o esqueleto do algoritmo de seleção de características.
        """
        mlflow.start_run()
        self.load_data()
        self.split_data()
        self.log_metrics_before_selection()
        self.compute_feature_scores()
        self.select_top_features()
        self.save_selected_features()
        self.log_metrics_after_selection()
        mlflow.end_run()