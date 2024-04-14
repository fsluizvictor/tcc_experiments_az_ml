import argparse
import os
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import xgboost as xgb

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score, precision_score, recall_score

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

def select_first_file(path):
    """Selects first file in folder, use under assumption there is only one file in folder
    Args:
        path (str): path to directory or file to choose
    Returns:
        str: full path of selected file
    """
    files = os.listdir(path)
    return os.path.join(path, files[0])

os.makedirs("./outputs", exist_ok=True)

def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="path to train data")
    parser.add_argument("--test_data", type=str, help="path to test data")
    
    # hyperparameters to GradientBoostingClassifier
    parser.add_argument("--n_estimators_to_gbc", required=False, default=100, type=int)
    parser.add_argument("--learning_rate_to_gbc", required=False, default=0.1, type=float)
    
    # hyperparameters to RandomForestClassifier
    parser.add_argument("--n_estimators_to_rfc", required=False, default=100, type=int)

    # hyperparameters to XGBoostClassifier
    parser.add_argument("--n_estimators_to_xgb", required=False, default=100, type=int)
    parser.add_argument("--learning_rate_to_xgb", required=False, default=0.1, type=float)
    
    # hyperparameters to SVC
    parser.add_argument("--kernel_to_svc", required=False, default='rbf', type=str)
    parser.add_argument("--gamma_to_svc", required=False, default='auto', type=str)

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
    
    
    models = [
        ("GradientBoostingClassifier",GradientBoostingClassifier(n_estimators=args.n_estimators_to_gbc, learning_rate=args.learning_rate_to_gbc)),
        ("RandomForestClassifier",RandomForestClassifier(n_estimators=args.n_estimators_to_rfc)),
        ("NaiveBayesClassifier", GaussianNB()),
        ("XGBoostClassifier", xgb.XGBClassifier(n_estimators=args.n_estimators_to_xgb, learning_rate=args.learning_rate_to_xgb)),
        ("SVC", SVC(kernel=args.kernel_to_svc, gamma=args.gamma_to_svc))
    ]
    
    for model_name, model in models:
        _train_pipeline(clf=model,
                        model_name=model_name,
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
    else:
        _sklearn_models(clf,
                    model_name,
                    path_to_model,
                    X_train,
                    X_test,
                    y_train,
                    y_test)



if __name__ == "__main__":
    main()