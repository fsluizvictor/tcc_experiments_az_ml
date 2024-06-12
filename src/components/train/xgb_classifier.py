import argparse
from base_train import BaseTrain
import xgboost as xgb
import mlflow
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

class XGBClassifier(BaseTrain):
    def train(self, n_estimators=100, learning_rate=0.1):

        mlflow.start_run()

        self.hyperparams.append(n_estimators)
        self.hyperparams.append(learning_rate)

        self.clf = xgb.XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate)

        self.clf.fit(self.x_train, self.y_train)
        self.y_pred = self.clf.predict(self.x_test)

        report = classification_report(self.y_test, self.y_pred)
        print(report)

        # Calculating metrics
        accuracy = accuracy_score(self.y_test, self.y_pred)
        f1 = f1_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred)
        recall = recall_score(self.y_test, self.y_pred)

        print(f"Metrics:accuracy: {accuracy},f1: {f1},precision: {precision},recall: {recall}")

        mlflow.log_metric('training_accuracy_score', accuracy)
        mlflow.log_metric('f1_score', f1)
        mlflow.log_metric('precision', precision)
        mlflow.log_metric('recall', recall)

        mlflow.log_metric("num_samples_x_train", self.x_train.shape[0])
        mlflow.log_metric("num_features_x_train", self.x_train.shape[1])

        mlflow.log_metric("num_samples_y_train", self.y_train.shape[0])
        mlflow.log_metric("num_features_y_train", self.y_train.shape[1])

        mlflow.log_metric("num_samples_x_test", self.x_test.shape[0])
        mlflow.log_metric("num_features_x_test", self.x_test.shape[1])

        mlflow.log_metric("num_samples_y_test", self.y_test.shape[0])
        mlflow.log_metric("num_features_y_test", self.y_test.shape[1])
        
        print("hyperparams", self.hyperparams)

        print("x_train", self.x_train)
        print("y_train", self.y_train)
        print("x_test", self.x_test)
        print("y_test", self.y_test)

        mlflow.end_run()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="path to train data")
    parser.add_argument("--test_data", type=str, help="path to test data")
    
    parser.add_argument("--n_estimators_to_xgb", required=False, default=100, type=int)
    parser.add_argument("--learning_rate_to_xgb", required=False, default=0.1, type=float)
    
    parser.add_argument("--registered_model_name", type=str, help="model name")
    parser.add_argument("--model", type=str, help="path to model file")
    
    args = parser.parse_args()

    classifier = XGBClassifier(
        model_name="XGBoostClassifier",
        train_data=args.train_data,
        test_data=args.test_data,
    )

    classifier.run(n_estimators=args.n_estimators_to_xgb, learning_rate=args.learning_rate_to_xgb)