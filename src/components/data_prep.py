import os
import argparse
import pandas as pd
import mlflow

def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_to_train", type=str, help="path to input data to train")
    parser.add_argument("--data_to_test", type=str, help="path to input data to test")
    parser.add_argument("--train_data", type=str, help="path to train data")
    parser.add_argument("--test_data", type=str, help="path to test data")
    parser.add_argument("--flag_remove_null_values", type=bool, help="flag to remove null values, add value when true, empty when false")
    parser.add_argument("--flag_remove_values_by_percentage", type=bool, help="flag to remove values with percentage")
    parser.add_argument("--percentage_to_remove_column", type=float, help="fpercentagr")

    args = parser.parse_args()

    # Start Logging
    mlflow.start_run()
    
    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    print("input data to train:", args.data_to_train)
    
    vrex_df_to_train = pd.read_csv(args.data_to_train)
    vrex_df_to_train = _remove_columns(vrex_df_to_train)

    mlflow.log_metric("num_samples_vrex_df_to_train_original", vrex_df_to_train.shape[0])
    mlflow.log_metric("num_features_vrex_df_to_train_original", vrex_df_to_train.shape[1] - 1)
    
    removed_columns = []

    if args.flag_remove_null_values and args.flag_remove_values_by_percentage and args.percentage_to_remove_column > 0:
        #Remove null values    
        vrex_df_to_train, removed_columns = _remove_values(vrex_df_to_train, args.percentage_to_remove_column)
        
        print("removed_columns", removed_columns)    

        mlflow.log_metric("num_samples_vrex_df_to_train_without_null_values", vrex_df_to_train.shape[0])
        mlflow.log_metric("num_features_vrex_df_to_train_without_null_values", vrex_df_to_train.shape[1] - 1)
    
    
    vrex_df_to_test = pd.read_csv(args.data_to_test)
    vrex_df_to_test = _remove_columns(vrex_df_to_test)    
    
    mlflow.log_metric("num_samples_vrex_df_to_test", vrex_df_to_test.shape[0])
    mlflow.log_metric("num_features_vrex_df_to_test", vrex_df_to_test.shape[1] - 1)
    
    if args.flag_remove_null_values and args.flag_remove_values_by_percentage and len(removed_columns) > 0:

        #Remove null values
        print("removed_columns", removed_columns)    
        vrex_df_to_test = vrex_df_to_test.drop(columns=removed_columns)
    
        mlflow.log_metric("num_samples_vrex_df_to_test_without_null_values", vrex_df_to_test.shape[0])
        mlflow.log_metric("num_features_vrex_df_to_test_without_null_values", vrex_df_to_test.shape[1] - 1)
    
    # output paths are mounted as folder, therefore, we are adding a filename to the path
    vrex_df_to_train.to_csv(os.path.join(args.train_data, "data.csv"), index=False)

    vrex_df_to_test.to_csv(os.path.join(args.test_data, "data.csv"), index=False)

    # Stop Logging
    mlflow.end_run()
    
    
def _remove_columns(df):
    return df.drop(columns=[  'lbl_exploits_delta_days',	
                        'lbl_exploits_weaponized_type_ENUM_absent',
                        'lbl_exploits_weaponized_type_ENUM_other',	
                        'lbl_exploits_weaponized_type_ENUM_auxiliary',
                        'lbl_exploits_weaponized_type_ENUM_exploit',
                        'lbl_exploits_weaponized_count',
                        'lbl_exploits_verified',
                        'idx',
                        'cve'
    ])
    
def _remove_values(df, threshold=0):
    all_columns = df.columns.tolist()
    df = df.loc[:, (df != 0).any(axis=0)]
    removed_columns = [column for column in all_columns if column not in df.columns]

    mlflow.log_metric("amount_columns_with_all_null_rows", len(removed_columns))
    print("columns with all null rows: ", len(removed_columns))

    # Calcula a porcentagem de valores nulos em cada coluna
    null_percentage = df.isnull().mean()

    # Seleciona as colunas que possuem uma porcentagem de valores nulos maior que o threshold
    columns_to_remove = null_percentage[null_percentage > threshold].index.tolist()

    mlflow.log_metric("percentage_to_remove_column", threshold)
    print("percentage to remove column: ", threshold)
    mlflow.log_metric("amount_columns_with_percentage", len(columns_to_remove))
    print("columns with percentage: ", len(columns_to_remove))

    # Remove as colunas selecionadas do DataFrame
    df = df.drop(columns=columns_to_remove)

    # Concatena as listas de colunas removidas devido a valores nulos e colunas removidas por outra razão
    all_removed_columns = removed_columns + columns_to_remove

    mlflow.log_metric("amount_removed_columns", len(all_removed_columns))
    print("amount removed columns: ", len(columns_to_remove))

    return df, all_removed_columns



if __name__ == "__main__":
    main()
