#!/usr/bin/env python3
import sys
sys.path.append('../')
import os
import pandas as pd
import source.get_baseline_features as get_features
import source.evaluation as evaluate
import wordfreq
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

get_features.preprocess()

data_location = os.path.join(os.path.dirname(os.getcwd()),"data")
train_loc = os.path.join(os.path.join(data_location,"train.csv"))
train_data = pd.read_csv(train_loc)
dev_loc = os.path.join(os.path.join(data_location,"dev.csv"))
dev_data = pd.read_csv(dev_loc)

train_features, train_fields = get_features.collect(train_data)
to_predict = ["FFDAvg","FFDStd","TRTAvg","TRTStd"]

val_features, val_fields = get_features.collect(dev_data)

MAE  = []
for feature in to_predict:
    model = SVR()
    model.fit(train_features, train_fields[feature])
    P_hat = model.predict(val_features)
    P = val_fields[feature]
    mae = evaluate.evaluate(P_hat,P)
    MAE.append(mae)

f_ = open("Results_Baseline_wordlen_logfreq_nextlog_prevlog","w")
print("Baseline MAE (SVR)")
print("Baseline MAE (SVR)",file=f_)
for field in range(len(MAE)):
    cat = to_predict[field]
    mae = MAE[field]
    print("%s (MAE)=\t%f"%(cat,mae))
    print("%s (MAE)=\t%f"%(cat,mae),file=f_)

MAE  = []
for feature in to_predict:
    model = LinearRegression()
    model.fit(train_features, train_fields[feature])
    P_hat = model.predict(val_features)
    P = val_fields[feature]
    mae = evaluate.evaluate(P_hat,P)
    MAE.append(mae)

print("Baseline MAE (LinearR)")
print("Baseline MAE (LinearR)",file=f_)
for field in range(len(MAE)):
    cat = to_predict[field]
    mae = MAE[field]
    print("%s (MAE)=\t%f"%(cat,mae))
    print("%s (MAE)=\t%f"%(cat,mae),file=f_)

