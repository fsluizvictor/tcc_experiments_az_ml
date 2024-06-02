import pandas as pd
import argparse
from sklearn.tree import DecisionTreeClassifier
from feature_selection.base_feature_selector import BaseFeatureSelector

class GiniFeatureSelection(BaseFeatureSelector):
    def compute_feature_scores(self):
        decision_tree = DecisionTreeClassifier(criterion='gini', random_state=42)
        decision_tree.fit(self.X_train, self.y_train)
        feature_importances = decision_tree.feature_importances_
        self.feature_scores = pd.DataFrame({'feature': self.X_train.columns, 'gini': feature_importances})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="path to input data to train")
    parser.add_argument("--test_data", type=str, help="path to input data to test")
    parser.add_argument("--train_data_feat_sel", type=str, help="path to train data")
    parser.add_argument("--test_data_feat_sel", type=str, help="path to test data")
    parser.add_argument("--feature_percentage", type=float, help="feature percentage")

    args = parser.parse_args()

    selector = GiniFeatureSelection(
        train_data=args.train_data,
        test_data=args.test_data,
        train_data_feat_sel=args.train_data_feat_sel,
        test_data_feat_sel=args.test_data_feat_sel,
        feature_percentage=args.feature_percentage,
        method="gini"
    )
    selector.run()

if __name__ == "__main__":
    main()