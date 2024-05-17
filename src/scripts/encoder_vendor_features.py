import pandas as pd
import numpy as np
from typing import List

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import VREX_PATH, FEATURE_TYPE, FEATURES_TO_MAINTAIN, NEW_FEATURE, NEW_FILE_PATH, CONCAT_VENDOR_FEATURES

def main():
    print("starting ...")
    df = _get_df(VREX_PATH) 
    df = _remove_columns(df=df)

    features = _extract_vendor_features(df=df, feature_type=FEATURE_TYPE, features_to_maintain=FEATURES_TO_MAINTAIN)

    df[NEW_FEATURE] = np.nan

    df = _build_new_feature(df=df, features=features)
    df = df.drop(columns=features)

    #df.to_csv(CONCAT_VENDOR_FEATURES, index=False)
    
    df = _encode_features_by_row(df=df)
    df.to_csv(NEW_FILE_PATH, index=False)

def _build_new_feature(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    features_by_row = []

    for idx, row in df.iterrows():
        for column_name, value in row.items():
            if column_name in features and value == 1:
                features_by_row.append(column_name)

        join_features_str = _join_encoders(features_by_row)

        df.loc[idx, NEW_FEATURE] = join_features_str
        features_by_row = []
        join_features_str = ''
    return df

def _encode_features_by_row(df: pd.DataFrame) -> pd.DataFrame:
    # Combine as palavras de todas as linhas em um único documento
    all_words = []

    # Percorra todas as linhas do DataFrame
    for idx, row in df.iterrows():
        tfidf_values = []

        # Obtenha as palavras na nova característica para a linha atual
        words = row[NEW_FEATURE]

        # Verifique se é uma string antes de tentar dividi-la
        if isinstance(words, str):
            # Adicione as palavras à lista
            all_words.extend(words.split(';'))

    # Una todas as palavras em uma única string separada por espaço
    all_documents = ' '.join(all_words)

    # Inicialize o vetorizador TF-IDF
    tfidf_vectorizer = TfidfVectorizer()

    # Aplique o vetorizador TF-IDF aos documentos
    tfidf_matrix = tfidf_vectorizer.fit_transform([all_documents])

    # Obtenha os nomes das características do vetorizador
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Para cada linha, calcule a média ponderada dos valores TF-IDF das palavras na nova característica
    for idx, row in df.iterrows():
        tfidf_values = []

        # Obtenha as palavras na nova característica para a linha atual
        words = row[NEW_FEATURE]

        # Verifique se é uma string antes de tentar dividi-la
        if isinstance(words, str):
            # Divida a string em palavras
            for word in words.split(';'):
                # Para cada palavra na nova característica, obtenha o valor TF-IDF correspondente
                try:
                    word_index = feature_names.index(word)
                    tfidf_value = tfidf_matrix[0, word_index]  # A matriz TF-IDF tem apenas uma linha (documento único)
                    tfidf_values.append(tfidf_value)
                except ValueError:
                    pass

            # Calcule a média ponderada dos valores TF-IDF
            if tfidf_values:
                weighted_average = sum(tfidf_values) / len(tfidf_values)
            else:
                weighted_average = 0

            # Atribua a média ponderada à nova coluna NEW_FEATURE para a linha atual
            df.loc[idx, NEW_FEATURE] = weighted_average

    return df

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

def _join_encoders(features_by_row: List[str]):
    return ';'.join(features_by_row)

if __name__ == "__main__":
    main()