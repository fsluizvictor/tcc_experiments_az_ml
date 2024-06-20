import argparse
import os
import pandas as pd

from train_utils import select_first_file, train_and_log_model, _log_inputs
from sklearn.naive_bayes import MultinomialNB

def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="path to train data")
    parser.add_argument("--test_data", type=str, help="path to test data")

    # hyperparameters 
    parser.add_argument("--alpha", required=False, default=0.1, type=float)
    parser.add_argument("--fit_prior", required=False, default=False, type=bool)
    
    parser.add_argument("--registered_model_name", type=str, help="model name")
    parser.add_argument("--model", type=str, help="path to model file")
    
    args = parser.parse_args()

    # paths are mounted as folder, therefore, we are selecting the file from folder
    train_df = pd.read_csv(select_first_file(args.train_data))

    # paths are mounted as folder, therefore, we are selecting the file from folder
    test_df = pd.read_csv(select_first_file(args.test_data))

    _log_inputs(train_df=train_df, test_df=test_df)

    TARGET = 'lbl_exploits_has'

    y_train = train_df.pop(TARGET)
    X_train = train_df.values
    y_test = test_df.pop(TARGET)
    X_test = test_df.values

    print(f"Training with data of shape {X_train.shape}")
    
    _train_pipeline(clf=MultinomialNB(alpha=args.alpha, 
                                      fit_prior=args.fit_prior),
                        model_name="MultinomialNBClassifier",
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