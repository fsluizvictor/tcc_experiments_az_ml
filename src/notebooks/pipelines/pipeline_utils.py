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