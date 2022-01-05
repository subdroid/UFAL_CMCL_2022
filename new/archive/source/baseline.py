#!/usr/bin/env python3

# Baseline model

import os
import pandas as pd
import wordfreq
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

def transform_features(data):
    Features = ["word_len","word_freq","prev_bi_freq","next_bi_freq"]
    print(data.shape)




data_location = os.path.join(os.path.dirname(os.getcwd()),"data")

train_loc = os.path.join(os.path.join(data_location,"train.csv"))
dev_loc = os.path.join(os.path.join(data_location,"dev.csv"))

train_data = pd.read_csv(train_loc)

dev_data = pd.read_csv(dev_loc)

"""Language Feature based regression. Experiment:1"""

Train_Features = transform_features(train_data.to_numpy())