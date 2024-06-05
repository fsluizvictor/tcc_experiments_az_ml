import pandas as pd
import argparse
from base_feature_selection import BaseFeatureSelection
from feature_selection_utils import SPEARMAN_FEAT_SEL, FEATURE_KEY

class SPEARMANFeatureSelection(BaseFeatureSelection):
    def compute_feature_scores(self):
        correlations = self.X_train.corrwith(self.y_train, method=SPEARMAN_FEAT_SEL) 
        self.feature_scores = pd.DataFrame({FEATURE_KEY: self.X_train.columns, SPEARMAN_FEAT_SEL: correlations.abs()})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="path to input data to train")
    parser.add_argument("--test_data", type=str, help="path to input data to test")
    parser.add_argument("--train_data_feat_sel", type=str, help="path to train data")
    parser.add_argument("--test_data_feat_sel", type=str, help="path to test data")
    parser.add_argument("--feature_percentage", type=float, help="feature percentage")

    args = parser.parse_args()

    selector = SPEARMANFeatureSelection(
        train_data=args.train_data,
        test_data=args.test_data,
        train_data_feat_sel=args.train_data_feat_sel,
        test_data_feat_sel=args.test_data_feat_sel,
        feature_percentage=args.feature_percentage,
        method_name=SPEARMAN_FEAT_SEL
    )
    selector.run()

if __name__ == "__main__":
    main()