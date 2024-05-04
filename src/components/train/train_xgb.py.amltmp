import argparse
import os
import pandas as pd

from train_utils import select_first_file, train_and_log_model
import xgboost as xgb

def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="path to train data")
    parser.add_argument("--test_data", type=str, help="path to test data")
    
    # hyperparameters to XGBoostClassifier
    parser.add_argument("--n_estimators_to_xgb", required=False, default=100, type=int)
    parser.add_argument("--learning_rate_to_xgb", required=False, default=0.1, type=float)
    
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
    
    _train_pipeline(clf=xgb.XGBClassifier(n_estimators=args.n_estimators_to_xgb, learning_rate=args.learning_rate_to_xgb),
                        model_name="XGBoostClassifier",
                        X_train=X_train,
                        X_test=X_test,
                        y_train=y_train,
                        y_test=y_test
                    )
        
def _train_pipeline(clf,
                    model_name,
                    X_train,
                    X_test,
                    y_train,
                    y_test):
    train_and_log_model(clf,
                    model_name,
                    X_train,
                    X_test,
                    y_train,
                    y_test)
    

if __name__ == "__main__":
    main()