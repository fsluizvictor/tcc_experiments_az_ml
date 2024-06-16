import pandas as pd
import argparse
from base_feature_selection import BaseFeatureSelection
from feature_selection_utils import PEARSON_FEAT_SEL, FEATURE_KEY

class PearsonFeatureSelection(BaseFeatureSelection):
    def compute_feature_scores(self):
        correlations = self.X_train.corrwith(self.y_train, method=PEARSON_FEAT_SEL) 
        self.feature_scores = pd.DataFrame({FEATURE_KEY: self.X_train.columns, PEARSON_FEAT_SEL: correlations.abs()})
        print("self.X_train.columns\n", self.X_train.columns)
        print("info_gain\n", correlations)
        print("self.feature_scores\n", self.feature_scores)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="path to input data to train")
    parser.add_argument("--test_data", type=str, help="path to input data to test")
    parser.add_argument("--train_data_feat_sel", type=str, help="path to train data")
    parser.add_argument("--test_data_feat_sel", type=str, help="path to test data")
    parser.add_argument("--feature_quantity", type=int, help="feature percentage")

    args = parser.parse_args()

    selector = PearsonFeatureSelection(
        train_data=args.train_data,
        test_data=args.test_data,
        train_data_feat_sel=args.train_data_feat_sel,
        test_data_feat_sel=args.test_data_feat_sel,
        feature_quantity=args.feature_quantity,
        method_name=PEARSON_FEAT_SEL
    )
    selector.run()

if __name__ == "__main__":
    main()