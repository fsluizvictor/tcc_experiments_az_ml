import argparse
from base_train import BaseTrain
import xgboost as xgb
import mlflow.xgboost

class XGBClassifier(BaseTrain):
    def train(self, n_estimators=100, learning_rate=0.1):
        mlflow.xgboost.autolog()

        self.hyperparams.append(n_estimators)
        self.hyperparams.append(learning_rate)

        self.clf = xgb.XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate)

        self.clf.fit(self.x_train, self.y_train)
        self.y_pred = self.clf.predict(self.x_test)

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