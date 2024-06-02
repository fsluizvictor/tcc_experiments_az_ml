import pandas as pd
import argparse
from sklearn.feature_selection import mutual_info_classif
from base_feature_selection import BaseFeatureSelection

class InfogainFeatureSelection(BaseFeatureSelection):
    def compute_feature_scores(self):
        info_gain = mutual_info_classif(self.X_train, self.y_train)
        self.feature_scores = pd.DataFrame({'Feature': self.X_train.columns, 'info_gain': info_gain})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="path to input data to train")
    parser.add_argument("--test_data", type=str, help="path to input data to test")
    parser.add_argument("--train_data_feat_sel", type=str, help="path to train data")
    parser.add_argument("--test_data_feat_sel", type=str, help="path to test data")
    parser.add_argument("--feature_percentage", type=float, help="feature percentage")

    args = parser.parse_args()

    selector = InfogainFeatureSelection(
        train_data=args.train_data,
        test_data=args.test_data,
        train_data_feat_sel=args.train_data_feat_sel,
        test_data_feat_sel=args.test_data_feat_sel,
        feature_percentage=args.feature_percentage,
        method="info_gain"
    )
    selector.run()

if __name__ == "__main__":
    main()