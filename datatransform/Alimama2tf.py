from transform_alimama import DataTransform
from datetime import datetime, date
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
parser = argparse.ArgumentParser(description='Transfrom original data to TFRecord')

#parser.add_argument('avazu', type=string)
parser.add_argument('--label', type=str, default="Label")
parser.add_argument("--store_stat", action="store_true", default=True)
parser.add_argument("--threshold", type=int, default=2)
parser.add_argument("--dataset1", type=Path, default='../data/train_data.csv')
parser.add_argument("--dataset2", type=Path, default='../data/test_data.csv')
parser.add_argument("--stats", type=Path, default='../data/stats_2')
parser.add_argument("--record", type=Path, default='../data/threshold_2')
parser.add_argument("--ratio", nargs='+', type=float, default=[0.9, 0.1, 0.0])

args = parser.parse_args()


class CoatTransform(DataTransform):
    def __init__(self, dataset_path1, dataset_path2, path, stats_path, min_threshold, label_index, ratio, store_stat=False, seed=2021):
        super(CoatTransform, self).__init__(dataset_path1, dataset_path2, stats_path, store_stat=store_stat, seed=seed)
        self.threshold = min_threshold
        self.label = label_index
        self.domain = 'domain_id'
        self.split = ratio
        self.path = path
        self.stats_path = stats_path
        self.name = ['U1', 'U2', 'U3', 'U4', 'U5', 'U6', 'U7', 'U8', 'I9', 'I10', 'I11', 'I12', 'I13', 'I_i14', 'C15', 'U_d16', 'Label', "domain_id"]

    def process(self):
        self._read(name=self.name, header=None, sep=",", label_index=self.label, domain_id=self.domain)
        if self.store_stat:
            white_list = ['I_i14']
            print('white_list len:', len(white_list))
            self.generate_and_filter(threshold=self.threshold, label_index=self.label, domain_id=self.domain, white_list=white_list)

        self.data = self.traindata
        tr, _, val = self.random_split(ratio=self.split)
        self.transform_tfrecord(tr, self.path, "train", label_index=self.label, domain_id=self.domain)
        self.transform_tfrecord(val, self.path, "validation", label_index=self.label, domain_id=self.domain)

        self.transform_tfrecord(self.testdata, self.path, "test", label_index=self.label, domain_id=self.domain)


    def _process_x(self):
        print(self.data[self.data["Label"] == 1].shape)

        def bucket(value):
            if not pd.isna(value):
                if value > 2:
                    value = int(np.floor(np.log(value) ** 2))
                else:
                    value = int(value)
            return value

        numeric_list = ['I_i14']
        for col_name in numeric_list:
            self.data[col_name] = self.data[col_name].apply(bucket)

    def _process_y(self):
        pass

if __name__ == "__main__":
    tranformer = CoatTransform(args.dataset1, args.dataset2, args.record, args.stats,
                                 args.threshold, args.label,
                                 args.ratio, store_stat=args.store_stat)
    tranformer.process()
