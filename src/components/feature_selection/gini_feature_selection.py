import pandas as pd
import argparse
from sklearn.tree import DecisionTreeClassifier
from base_feature_selection import BaseFeatureSelection
from feature_selection_utils import GINI_FEAT_SEL, FEATURE_KEY

class GiniFeatureSelection(BaseFeatureSelection):
    def compute_feature_scores(self):
        decision_tree = DecisionTreeClassifier(criterion=GINI_FEAT_SEL, random_state=42)
        decision_tree.fit(self.X_train, self.y_train)
        feature_importances = decision_tree.feature_importances_
        self.feature_scores = pd.DataFrame({FEATURE_KEY: self.X_train.columns, GINI_FEAT_SEL: feature_importances})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="path to input data to train")
    parser.add_argument("--test_data", type=str, help="path to input data to test")
    parser.add_argument("--train_data_feat_sel", type=str, help="path to train data")
    parser.add_argument("--test_data_feat_sel", type=str, help="path to test data")
    parser.add_argument("--feature_quantity", type=float, help="feature percentage")

    args = parser.parse_args()

    selector = GiniFeatureSelection(
        train_data=args.train_data,
        test_data=args.test_data,
        train_data_feat_sel=args.train_data_feat_sel,
        test_data_feat_sel=args.test_data_feat_sel,
        feature_quantity=args.feature_quantity,
        method_name=GINI_FEAT_SEL
    )
    selector.run()

if __name__ == "__main__":
    main()