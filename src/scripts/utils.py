#encoder_vendor_features.py
VREX_PATH = '/home/luiz/repos/tcc_experiments_az_ml/data/vrex.csv'
NEW_FILE_PATH = '/home/luiz/repos/tcc_experiments_az_ml/data/vrex_encoder_vendors_tfidf.csv'
CONCAT_VENDOR_FEATURES = '/home/luiz/repos/tcc_experiments_az_ml/data/vrex_concatenated_vendors.csv'

FEATURE_TYPE = 'vendor_ENUM'
FEATURES_TO_MAINTAIN = ['vendor_ENUM_absent']
NEW_FEATURE = ['vendor_ENUM_encoded']

#batch_datasets.py
RANGE_YEARS_TRAIN = (2008,2017)
RANGE_YEARS_TEST = (2018,2021)

VREX_ENCODER_VENDORS_TFIDF = '/home/luiz/repos/tcc_experiments_az_ml/data/vrex_encoder_vendors_tfidf.csv'
NEW_FILE_BASE_PATH = '/home/luiz/repos/tcc_experiments_az_ml/data/samples/modified/vrex_vendor_tf_idf_'