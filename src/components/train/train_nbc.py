import argparse
import os
import pandas as pd

from utils.train_utils import select_first_file, sklearn_models
from sklearn.naive_bayes import GaussianNB

def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="path to train data")
    parser.add_argument("--test_data", type=str, help="path to test data")
    
    parser.add_argument("--registered_model_name", type=str, help="model name")
    parser.add_argument("--model", type=str, help="path to model file")
    
    args = parser.parse_args()

    # paths are mounted as folder, therefore, we are selecting the file from folder
    train_df = pd.read_csv(select_first_file(args.train_data))

    # paths are mounted as folder, therefore, we are selecting the file from folder
    test_df = pd.read_csv(select_first_file(args.test_data))

    TARGET = 'lbl_exploits_has'

    y_train = train_df.pop(TARGET)
    X_train = train_df.values
    y_test = test_df.pop(TARGET)
    X_test = test_df.values

    print(f"Training with data of shape {X_train.shape}")
    
    _train_pipeline(clf=GaussianNB(),
                        model_name="NaiveBayesClassifier",
                        path_to_model=args.model,
                        X_train=X_train,
                        X_test=X_test,
                        y_train=y_train,
                        y_test=y_test
                    )
        
def _train_pipeline(clf,
                    model_name,
                    path_to_model,
                    X_train,
                    X_test,
                    y_train,
                    y_test):
    sklearn_models(clf,
                    model_name,
                    path_to_model,
                    X_train,
                    X_test,
                    y_train,
                    y_test)

if __name__ == "__main__":
    main()