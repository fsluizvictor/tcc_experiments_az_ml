from typing import List
import datetime as dt

#SWEEP VALUES
TOTAL_TRIALS = 20
CONCURRENT_TRIALS = 10
TIMEOUT = 2*3600
TIMEOUT_PLUS = 2*7200
COMPUTE = 'serverless'
SAMPLING_ALGORITHM = 'bayesian'
METRIC = 'training_accuracy_score'
GOAL = 'Maximize'
EVALUATION_INTERVAL = 1
DELAY_EVALUATION = 5

#AUTHENTICATE VALUES
SUBSCRIPTION = "1f51a68b-69d8-4818-bf63-7c5e10b81967"
RESOURCE_GROUP = "luiz.victor.dev-rg"
WS_NAME = "tcc-tests"

#COMPONENTS PATH
#TRAIN
GBC_PATH = "/home/azureuser/cloudfiles/code/Users/luiz.victor.dev/tcc_experiments_az_ml/src/components/train/train_gbc.yaml"
NBC_PATH = "/home/azureuser/cloudfiles/code/Users/luiz.victor.dev/tcc_experiments_az_ml/src/components/train/train_nbc.yaml"
MULT_NBC_PATH = "/home/azureuser/cloudfiles/code/Users/luiz.victor.dev/tcc_experiments_az_ml/src/components/train/train_multinomial_nbc.yaml"
RFC_PATH = "/home/azureuser/cloudfiles/code/Users/luiz.victor.dev/tcc_experiments_az_ml/src/components/train/train_rfc.yaml"
SVC_PATH = "/home/azureuser/cloudfiles/code/Users/luiz.victor.dev/tcc_experiments_az_ml/src/components/train/train_svc.yaml"
XGB_PATH = "/home/azureuser/cloudfiles/code/Users/luiz.victor.dev/tcc_experiments_az_ml/src/components/train/train_xgb.yaml"

#FEAT_SEL
INFOGAIN_PATH = "/home/azureuser/cloudfiles/code/Users/luiz.victor.dev/tcc_experiments_az_ml/src/components/feature_selection/infogain_feature_selection.yaml"
GINI_PATH = "/home/azureuser/cloudfiles/code/Users/luiz.victor.dev/tcc_experiments_az_ml/src/components/feature_selection/gini_feature_selection.yaml"
SPEARMAN_PATH = "/home/azureuser/cloudfiles/code/Users/luiz.victor.dev/tcc_experiments_az_ml/src/components/feature_selection/spearman_feature_selection.yaml"
PEARSON_PATH = "/home/azureuser/cloudfiles/code/Users/luiz.victor.dev/tcc_experiments_az_ml/src/components/feature_selection/pearson_feature_selection.yaml"

#PREP_DATA
PREP_DATA_PATH = "/home/azureuser/cloudfiles/code/Users/luiz.victor.dev/tcc_experiments_az_ml/src/components/preparation/data_prep.yaml"

#FEAT SEL
GINI = 'Gini'
INFOGAIN = 'Infogain'
PEARSON = 'Pearson'
SPEARMAN = 'spearman'

#TRAIN
GBC = 'GradientBoostingClassifier'
NBC = 'GaussianNB'
RFC = 'RandomForestClassifier'
SVC = 'SVC'
XGB = 'XGBoost'
MODELS = [NBC, GBC, RFC, SVC, XGB]

#PREPATION
DATA_PREP = 'DataPreparation'

#PIPELINES BY FEATURE SELECTION
#INFOGAIN
GBC_BY_INFOGAIN = f"{GBC}_BY_{INFOGAIN}"
NBC_BY_INFOGAIN = f"{NBC}_BY_{INFOGAIN}"
RFC_BY_INFOGAIN = f"{RFC}_BY_{INFOGAIN}"
SVC_BY_INFOGAIN = f"{SVC}_BY_{INFOGAIN}"
XGB_BY_INFOGAIN = f"{XGB}_BY_{INFOGAIN}"

#GINI
GBC_BY_GINI = f"{GBC}_BY_{GINI}"
NBC_BY_GINI = f"{NBC}_BY_{GINI}"
MULT_NBC_BY_GINI = f"{NBC}_BY_{GINI}"
RFC_BY_GINI = f"{RFC}_BY_{GINI}"
SVC_BY_GINI = f"{SVC}_BY_{GINI}"
XGB_BY_GINI = f"{XGB}_BY_{GINI}"

#SPEARMAN
GBC_BY_SPEARMAN = f"{GBC}_BY_{SPEARMAN}"
NBC_BY_SPEARMAN = f"{NBC}_BY_{SPEARMAN}"
RFC_BY_SPEARMAN = f"{RFC}_BY_{SPEARMAN}"
SVC_BY_SPEARMAN = f"{SVC}_BY_{SPEARMAN}"
XGB_BY_SPEARMAN = f"{XGB}_BY_{SPEARMAN}"

#PEARSON
GBC_BY_PEARSON = f"{GBC}_BY_{PEARSON}"
NBC_BY_PEARSON = f"{NBC}_BY_{PEARSON}"
RFC_BY_PEARSON = f"{RFC}_BY_{PEARSON}"
SVC_BY_PEARSON = f"{SVC}_BY_{PEARSON}"
XGB_BY_PEARSON = f"{XGB}_BY_{PEARSON}"

#FUNCTIONS

def get_experiment_names(train_data: List[str], test_data: List[str], feat_sel: str, n_features: List[float]) -> List[str]:
    experiment_names = []
    for n_feature in n_features:
        for train_name, test_name in zip(train_data, test_data):
            for model_name in MODELS:
                current_time = dt.datetime.now()
                formatted_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")  # Formata a data e hora atual
                train_name_base = train_name.split('.')[0]
                test_name_base = test_name.split('.')[0]
                name = f"{train_name_base}_tested_{test_name_base}_{feat_sel}_{model_name}_{formatted_time}"
                experiment_names.append(name)
                print(name)
    return experiment_names