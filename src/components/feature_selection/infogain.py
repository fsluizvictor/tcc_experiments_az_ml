import os
import argparse
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import mlflow

def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="path to input data to train")
    parser.add_argument("--test_data", type=str, help="path to input data to test")
    parser.add_argument("--train_data_feat_sel", type=str, help="path to train data")
    parser.add_argument("--test_data_feat_sel", type=str, help="path to test data")
    parser.add_argument("--feature_percentage", type=int, help="feature percentage")

    args = parser.parse_args()

    # Start Logging
    mlflow.start_run()

    print("start infogain.py ...")
    
    df_train = pd.read_csv(args.train_data)
    df_test = pd.read_csv(args.test_data)

    X_train = df_train
    X_test = df_test 

    TARGET = 'lbl_exploits_has'

    y_train = X_train.pop(TARGET)
    X_train = X_train.values
    y_test = X_test.pop(TARGET)
    X_test = X_test.values

    mlflow.log_metric("num_samples_train_original", df_train.shape[0])
    mlflow.log_metric("num_features_train_original", df_train.shape[1] - 1)
    mlflow.log_metric("num_samples_test_original", df_test.shape[0])
    mlflow.log_metric("num_features_test_original", df_test.shape[1] - 1)

    info_gain = mutual_info_classif(X_train, y_train)

    feature_scores = pd.DataFrame({'Feature': X.columns, 'Information_Gain': info_gain})
    feature_scores = feature_scores.sort_values(by='Information_Gain', ascending=False)

    percentage = args.feature_percentage

    feature_quantity = (X_train.shape[1] - 1) * percentage

    mlflow.log_metric("feature_percentage", percentage)
    mlflow.log_metric("feature_quantity", feature_quantity)
    print(f"feature_percentage: {percentage}, feature_quantity: {feature_quantity}")

    top_features = feature_scores.head(feature_quantity)

    mlflow.log_metric("top_features", top_features)
    print("top_features:", top_features)

    df_train_selected = df_train[top_features['Feature']]
    df_test_selected = df_test[top_features['Feature']]

    mlflow.log_metric("num_samples_train_feat_sel", df_train.shape[0])
    mlflow.log_metric("num_features_train_feat_sel", df_train.shape[1] - 1)
    mlflow.log_metric("num_samples_test_feat_sel", df_test.shape[0])
    mlflow.log_metric("num_features_test_feat_sel", df_test.shape[1] - 1)

    # output paths are mounted as folder, therefore, we are adding a filename to the path
    df_train_selected.to_csv(os.path.join(args.train_data_feat_sel, "data.csv"), index=False)

    df_test_selected.to_csv(os.path.join(args.test_data_feat_sel, "data.csv"), index=False)

    # Stop Logging
    mlflow.end_run()
    
if __name__ == "__main__":
    main()
