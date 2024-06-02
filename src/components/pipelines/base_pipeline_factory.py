from abc import ABC, abstractmethod
from typing import Callable
from azure.ai.ml import Input

class BasePipelineFactory(ABC):
    def __init__(self, data_prep_component: Callable, feature_selection_component: Callable, train_component : Callable):
        self.data_prep_component = data_prep_component
        self.feature_selection_component = feature_selection_component
        self.train_component = train_component

    @abstractmethod
    def create_pipeline(self, data_to_train: Input, data_to_test: Input):
        pass
