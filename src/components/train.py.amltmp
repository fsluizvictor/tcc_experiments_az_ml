import argparse
import os
import pandas as pd
import mlflow
import mlflow.sklearn
import xgboost as xgb

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_fscore_support

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
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
    parser.add_argument("--n_estimators", required=False, default=100, type=int)
    parser.add_argument("--learning_rate", required=False, default=0.1, type=float)
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
        ("GradientBoostingClassifier",GradientBoostingClassifier(n_estimators=args.n_estimators, learning_rate=args.learning_rate)),
        ("RandomForestClassifier",RandomForestClassifier(n_estimators=args.n_estimators)),
        ("NaiveBayesClassifier", GaussianNB()),
        ("XGBoostClassifier", xgb.XGBClassifier(n_estimators=args.n_estimators, learning_rate=args.learning_rate)),
        ("SVC", SVC(kernel='rbf', gamma='auto'))
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
    _is_active()
    print(f"Training with data of shape {X_train.shape}")
    mlflow.start_run()
    mlflow.sklearn.autolog()
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred))

    mlflow.sklearn.log_model(
        sk_model=clf,
        artifact_path=model_name,
    )

    mlflow.end_run()
    
def _is_active():
    if mlflow.active_run():
        mlflow.end_run()

if __name__ == "__main__":
    main()