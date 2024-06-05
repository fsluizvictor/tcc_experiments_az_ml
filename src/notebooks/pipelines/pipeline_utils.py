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
SUBSCRIPTION = "a3f56f48-3efb-4970-81a3-e4eda598333c"
RESOURCE_GROUP = "luiz.victor.dev-rg"
WS_NAME = "tcc-experiments"

#COMPONENTS PATH
#TRAIN
GBC_PATH = "/home/azureuser/cloudfiles/code/Users/luiz.victor.dev/tcc_experiments_az_ml/src/components/train/train_gbc.yaml"
NBC_PATH = "/home/azureuser/cloudfiles/code/Users/luiz.victor.dev/tcc_experiments_az_ml/src/components/train/train_nbc.yaml"
RFC_PATH = "/home/azureuser/cloudfiles/code/Users/luiz.victor.dev/tcc_experiments_az_ml/src/components/train/train_rfc.yaml"
SVC_PATH = "/home/azureuser/cloudfiles/code/Users/luiz.victor.dev/tcc_experiments_az_ml/src/components/train/train_svc.yaml"
XGB_PATH = "/home/azureuser/cloudfiles/code/Users/luiz.victor.dev/tcc_experiments_az_ml/src/components/train/train_xgb.yaml"

#FEAT_SEL
INFOGAIN_PATH = "/home/azureuser/cloudfiles/code/Users/luiz.victor.dev/tcc_experiments_az_ml/src/components/feature_selection/infogain_feature_selection.yaml"
GINI_PATH = "/home/azureuser/cloudfiles/code/Users/luiz.victor.dev/tcc_experiments_az_ml/src/components/feature_selection/gini_feature_selection.yaml"
SPERMAN_PATH = "/home/azureuser/cloudfiles/code/Users/luiz.victor.dev/tcc_experiments_az_ml/src/components/feature_selection/sperman_feature_selection.yaml"
PEARSON_PATH = "/home/azureuser/cloudfiles/code/Users/luiz.victor.dev/tcc_experiments_az_ml/src/components/feature_selection/pearson_feature_selection.yaml"

#PREP_DATA
PREP_DATA_PATH = "/home/azureuser/cloudfiles/code/Users/luiz.victor.dev/tcc_experiments_az_ml/src/components/preparation/data_prep.yaml"

#FEAT SEL
GINI = 'Gini'
INFOGAIN = 'Infogain'
PEARSON = 'Pearson'
SPERMAN = 'Sperman'

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
RFC_BY_GINI = f"{RFC}_BY_{GINI}"
SVC_BY_GINI = f"{SVC}_BY_{GINI}"
XGB_BY_GINI = f"{XGB}_BY_{GINI}"

#SPERMAN
GBC_BY_SPERMAN = f"{GBC}_BY_{SPERMAN}"
NBC_BY_SPERMAN = f"{NBC}_BY_{SPERMAN}"
RFC_BY_SPERMAN = f"{RFC}_BY_{SPERMAN}"
SVC_BY_SPERMAN = f"{SVC}_BY_{SPERMAN}"
XGB_BY_SPERMAN = f"{XGB}_BY_{SPERMAN}"

#PEARSON
GBC_BY_PEARSON = f"{GBC}_BY_{PEARSON}"
NBC_BY_PEARSON = f"{NBC}_BY_{PEARSON}"
RFC_BY_PEARSON = f"{RFC}_BY_{PEARSON}"
SVC_BY_PEARSON = f"{SVC}_BY_{PEARSON}"
XGB_BY_PEARSON = f"{XGB}_BY_{PEARSON}"

