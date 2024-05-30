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

    df.to_csv(CONCAT_VENDOR_FEATURES, index=False)
    
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
    # Inicialize o vetorizador TF-IDF
    tfidf_vectorizer = TfidfVectorizer()

    print(f"Conteúdo da nova característica antes de TF-IDF:\n{df[NEW_FEATURE].head()}")

    # Crie a lista de todas as características em todas as linhas
    all_documents = []
    for idx, row in df.iterrows():
        words = row[NEW_FEATURE].values[0]
        if isinstance(words, str):
            words_list = words.split(';')
            words_list = [word.lower() for word in words_list]
            all_documents.extend(words_list)
    
    print(f"Documentos combinados para TF-IDF:\n{all_documents[:5]}")

    if not all_documents:
        raise ValueError("Nenhum documento válido encontrado para a vetorização TF-IDF.")

    # Ajuste a lista para o formato esperado pelo TF-IDF
    all_documents = [' '.join(all_documents)]

    tfidf_vectorizer.fit(all_documents)

    # Obtenha os valores TF-IDF para todas as palavras
    tfidf_matrix = tfidf_vectorizer.transform(all_documents)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Para cada linha, calcule a média ponderada dos valores TF-IDF das palavras na nova característica
    for idx, row in df.iterrows():
        words = row[NEW_FEATURE].values[0]
        if not words == '':
            words_list = words.split(';')
            words_list = [word.lower() for word in words_list]

            # Calcule os valores TF-IDF das palavras na linha
            word_tfidf_values = []
            for word in words_list:
                if word in feature_names:
                    index = feature_names.tolist().index(word)
                    tfidf_value = tfidf_matrix[0, index]
                    weighted_value = tfidf_value
                    word_tfidf_values.append(weighted_value)
                    print(f"Palavra: {word}, TF-IDF: {tfidf_value}")

            # Calcule a média ponderada dos valores TF-IDF das palavras presentes na linha
            weighted_average = np.sum(word_tfidf_values)
            print(f"Soma do TF-IDF para a linha {idx}: {weighted_average}")

            words_list = []
            word_tfidf_values = []

            df.loc[idx, NEW_FEATURE] = weighted_average
        else:
            df.loc[idx, NEW_FEATURE] = 0
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
    ])

def _join_encoders(features_by_row: List[str]):
    return ';'.join(features_by_row)

if __name__ == "__main__":
    main()