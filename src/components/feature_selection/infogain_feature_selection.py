import pandas as pd
import argparse
from sklearn.feature_selection import mutual_info_classif
from base_feature_selection import BaseFeatureSelection
from feature_selection_utils import INFOGAIN_FEAT_SEL, FEATURE_KEY

class InfogainFeatureSelection(BaseFeatureSelection):
    def compute_feature_scores(self):
        info_gain = mutual_info_classif(self.X_train, self.y_train)
        self.feature_scores = pd.DataFrame({FEATURE_KEY: self.X_train.columns, INFOGAIN_FEAT_SEL: info_gain})
        print("self.X_train.columns\n", self.X_train.columns)
        print("info_gain\n", info_gain)
        print("self.feature_scores\n", self.feature_scores)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="path to input data to train")
    parser.add_argument("--test_data", type=str, help="path to input data to test")
    parser.add_argument("--train_data_feat_sel", type=str, help="path to train data")
    parser.add_argument("--test_data_feat_sel", type=str, help="path to test data")
    parser.add_argument("--feature_quantity", type=int, help="feature percentage")

    args = parser.parse_args()

    selector = InfogainFeatureSelection(
        train_data=args.train_data,
        test_data=args.test_data,
        train_data_feat_sel=args.train_data_feat_sel,
        test_data_feat_sel=args.test_data_feat_sel,
        feature_quantity=args.feature_quantity,
        method_name=INFOGAIN_FEAT_SEL
    )
    selector.run()

if __name__ == "__main__":
    main()