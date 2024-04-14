import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report


def sklearn_models(clf,
                    model_name,
                    X_train,
                    X_test,
                    y_train,
                    y_test):
    _is_active()
    print(f"Training with data of shape {X_train.shape}")
    
    mlflow.start_run(run_name=model_name)

    mlflow.sklearn.autolog()
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred))

    # Metric to hyperparameter
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric('Accuracy', float(accuracy))

    mlflow.end_run()

def xgboost_models(clf,
                    model_name,
                    X_train,
                    X_test,
                    y_train,
                    y_test):
    _is_active()
    print("running: ", model_name)
    print(f"Training with data of shape {X_train.shape}")
    
    mlflow.autolog()
    
    with mlflow.start_run(run_name=model_name):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Calculating metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        # Alternatively, you can use clf.score() if it's available for your model

        # Metric to hyperparameter
        mlflow.log_metric('Accuracy', float(accuracy))

        # Logging metrics
        mlflow.log_metric("training_accuracy_score", accuracy)
        mlflow.log_metric("training_f1_score", f1)
        mlflow.log_metric("training_precision_score", precision)
        mlflow.log_metric("training_recall_score", recall)
        
        # You can also log the overall training score if available
        if hasattr(clf, 'score'):
            training_score = clf.score(X_train, y_train)
            mlflow.log_metric("training_score", training_score)

        mlflow.end_run()

    
def _is_active():
    if mlflow.active_run():
        mlflow.end_run()