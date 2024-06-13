import os
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

def select_first_file(path) -> str:
    """Selecione o primeiro arquivo em uma pasta, assumindo que há apenas um arquivo na pasta.
    
    Args:
        path (str): Caminho para o diretório ou arquivo a ser escolhido.
        
    Returns:
        str: Caminho completo do arquivo selecionado.
    """
    files = os.listdir(path)
    return os.path.join(path, files[0])

os.makedirs("./outputs", exist_ok=True)

def train_and_log_model(clf,
                    model_name,
                    X_train,
                    X_test,
                    y_train,
                    y_test):
    """Treina o modelo e registra as métricas no MLflow.
    
    Args:
        clf: Classificador ou regressor a ser treinado.
        model_name (str): Nome do modelo para o run do MLflow.
        X_train (array-like): Dados de treinamento de entrada.
        y_train (array-like): Rótulos de treinamento.
        X_test (array-like): Dados de teste de entrada.
        y_test (array-like): Rótulos de teste.
    """
    print(f"Start train to {model_name}")
    _is_active()
    _validate_inputs(X_train, X_test, y_train, y_test)

    print(f"Training with data of shape {X_train.shape}")

    if model_name != "XGBoostClassifier":
        with mlflow.start_run(run_name=model_name):
            mlflow.sklearn.autolog()

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            return
    
    with mlflow.start_run(run_name=model_name):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        report = classification_report(y_test, y_pred)
        print(report)
            # Calculating metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        mlflow.log_metric('training_accuracy_score', accuracy)
        mlflow.log_metric('f1_score', f1)
        mlflow.log_metric('precision', precision)
        mlflow.log_metric('recall', recall)

        print(f"Metrics:accuracy: {accuracy},f1: {f1},precision: {precision},recall: {recall}")
        
        mlflow.end_run()

def _validate_inputs(X_train, X_test, y_train, y_test):
    """Verifica se os dados de entrada têm o formato correto.
    
    Args:
        X_train (array-like): Dados de treinamento de entrada.
        X_test (array-like): Dados de teste de entrada.
        y_train (array-like): Rótulos de treinamento.
        y_test (array-like): Rótulos de teste.
        
    Raises:
        ValueError: Se os dados de entrada não forem consistentes ou contiverem valores inválidos.
    """
    
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("Os tamanhos dos conjuntos de entrada e saída devem ser iguais.")
    
    mlflow.log_metric("num_samples_x_train", X_train.shape[0])
    mlflow.log_metric("num_features_x_train", X_train.shape[1])

    mlflow.log_metric("num_samples_y_train", y_train.shape[0])
    if len(y_train.shape) > 1:
        mlflow.log_metric("num_features_y_train", y_train.shape[1])
    else:
        mlflow.log_metric("num_features_y_train", 1)

    mlflow.log_metric("num_samples_x_test", X_test.shape[0])
    mlflow.log_metric("num_features_x_test", X_test.shape[1])

    mlflow.log_metric("num_samples_y_test", y_test.shape[0])
    if len(y_test.shape) > 1:
        mlflow.log_metric("num_features_y_test", y_test.shape[1])
    else:
        mlflow.log_metric("num_features_y_test", 1)
        
    print("x_train", X_train)
    print("y_train", y_train)
    print("x_test", X_test)
    print("y_test", y_test)
    
def _is_active():
    """Encerra o run ativo do MLflow, se houver."""
    if mlflow.active_run():
        mlflow.end_run()
