import pandas as pd
import numpy as np
from typing import List

from sklearn.preprocessing import LabelEncoder

from utils import VREX_PATH, FEATURE_TYPE, FEATURES_TO_MAINTAIN, NEW_FEATURE, NEW_FILE_PATH

def main():
    print("starting ...")
    df = _get_df(VREX_PATH) 
    df = _remove_columns(df=df)
    print("step 1 - extract features")
    features = _extract_vendor_features(df=df, feature_type=FEATURE_TYPE, features_to_maintain=FEATURES_TO_MAINTAIN)
    print("step 2 - apply encoder")
    encondeds = _apply_encoder(features)
    print("step 3 - dict of feature by encode")
    pairs = [(feature, encoder) for feature, encoder in zip(features, encondeds)]
    pairs = dict(pairs)

    df[NEW_FEATURE] = np.nan

    print("step 4 - register by encoder in rows")
    
    df = _encode_features_by_row(df=df)
    
    df = df.drop(columns=features)

    print("step 5 - new file")
    
    df.to_csv(NEW_FILE_PATH, index=False)

def _encode_features_by_row(df: pd.DataFrame) -> pd.DataFrame:
    encoders_by_row = []
    for idx, row in df.iterrows():
        for column_name, value in row.items():
            if column_name in features and value == 1:
                encoder = pairs[column_name]
                encoders_by_row.append(encoder)
        df.loc[idx, NEW_FEATURE] = _join_encoders(encoders_by_row)
        encoders_by_row = []
    return df

def _apply_encoder(features: List[str]):
    le = LabelEncoder()
    le.fit(features)
    encoded = le.transform(features)
    return encoded

def _get_df(path:str):
    return pd.read_csv(path)

def _extract_vendor_features(df: pd.DataFrame, feature_type: str, features_to_maintain: List[str]) -> List[str]:
    return [column_name for column_name in df.columns if feature_type in column_name and column_name not in features_to_maintain]

def _remove_columns(df : pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[  'lbl_exploits_delta_days',	
                        'lbl_exploits_weaponized_type_ENUM_absent',
                        'lbl_exploits_weaponized_type_ENUM_other',	
                        'lbl_exploits_weaponized_type_ENUM_auxiliary',
                        'lbl_exploits_weaponized_type_ENUM_exploit',
                        'lbl_exploits_weaponized_count',
                        'lbl_exploits_verified',
                        'idx',
                        'cve',
                        'lbl_exploits_has'
    ])

def _join_encoders(encoders_by_row):
    return ';'.join(map(str, encoders_by_row))

if __name__ == "__main__":
    main()