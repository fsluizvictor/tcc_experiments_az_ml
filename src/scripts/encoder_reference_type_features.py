import pandas as pd
import numpy as np
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer

from utils import VREX_ENCODER_VENDORS_TFIDF_TO_REF_TYPE, FEATURE_TYPE_REFERENCE_TYPE_ENUM, FEATURES_TO_MAINTAIN, CONCAT_REFERENCE_TYPE_FEATURE, NEW_FILE_PATH, CONCAT_VENDOR_FEATURES, VREX_ENCODER_TFIDF, CONCAT_REFERENCES_FEATURES

def main():
    print("starting ...")
    df = _get_df(VREX_ENCODER_VENDORS_TFIDF_TO_REF_TYPE) 

    features = _extract_reference_type_features(df=df, feature_type=FEATURE_TYPE_REFERENCE_TYPE_ENUM)

    df[CONCAT_REFERENCE_TYPE_FEATURE] = np.nan

    df = _build_new_feature(df=df, features=features)
    df = df.drop(columns=features)

    df.to_csv(CONCAT_REFERENCES_FEATURES, index=False)
    
    df = _encode_features_by_row(df=df)
    df.to_csv(VREX_ENCODER_TFIDF, index=False)

def _build_new_feature(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    features_by_row = []

    for idx, row in df.iterrows():
        for column_name, value in row.items():
            if column_name in features and value == 1:
                features_by_row.append(column_name)

        join_features_str = _join_encoders(features_by_row)

        df.loc[idx, CONCAT_REFERENCE_TYPE_FEATURE] = join_features_str
        features_by_row = []
        join_features_str = ''
    return df

def _encode_features_by_row(df: pd.DataFrame) -> pd.DataFrame:
    # Inicialize o vetorizador TF-IDF
    tfidf_vectorizer = TfidfVectorizer()

    print(f"Conteúdo da nova característica antes de TF-IDF:\n{df[CONCAT_REFERENCE_TYPE_FEATURE].head()}")

    # Crie a lista de todas as características em todas as linhas
    all_documents = []
    for idx, row in df.iterrows():
        words = row[CONCAT_REFERENCE_TYPE_FEATURE]
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
        words = row[CONCAT_REFERENCE_TYPE_FEATURE]
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

            weighted_average = np.sum(word_tfidf_values)
            print(f"Soma do TF-IDF para a linha {idx}: {weighted_average}")

            words_list = []
            word_tfidf_values = []

            df.loc[idx, CONCAT_REFERENCE_TYPE_FEATURE] = weighted_average
        else:
            df.loc[idx, CONCAT_REFERENCE_TYPE_FEATURE] = 0.0
    return df

def _get_df(path:str):
    return pd.read_csv(path)

def _extract_reference_type_features(df: pd.DataFrame, feature_type: str) -> List[str]:
    return [column_name for column_name in df.columns if feature_type in column_name]

def _join_encoders(features_by_row: List[str]):
    return ';'.join(features_by_row)

if __name__ == "__main__":
    main()