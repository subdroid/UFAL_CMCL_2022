#!/usr/bin/env python3

"""
Version:1.0
"""
import pandas as pd
import os
import numpy as np
import random


"""Reader for experiment 1. Still unsure about how to pass the data :("""
# def reader(data):
#     d_ = pd.read_csv(data)
#     words = d_["word"].tolist()
#     FFDAvg = d_["FFDAvg"].tolist()
#     FFDStd = d_["FFDStd"].tolist()
#     TRTAvg = d_["TRTAvg"].tolist()
#     TRTStd = d_["TRTStd"].tolist()
    
#     return words,FFDAvg,FFDStd,TRTAvg,TRTStd


def reader(data):
    d_ = pd.read_csv(data)
    sentences = np.unique(d_["sentence_id"])
    sents = []
    sentencs = []
    FFDAvg = []
    FFDStd = [] 
    TRTAvg = []
    TRTStd = []
    
    for sent in sentences:
        el = d_.loc[d_['sentence_id'] == sent]
        sents.append([el["word"].tolist(),el["FFDAvg"].tolist(),el["FFDStd"].tolist(),el["TRTAvg"].tolist(),el["TRTStd"].tolist()])
    # random.shuffle(sents)
    
    for sent in sents:
        sentencs.append(sent[0])
        FFDAvg.append(sent[1])
        FFDStd.append(sent[2])
        TRTAvg.append(sent[3])
        TRTStd.append(sent[4])
    return sentencs,FFDAvg,FFDStd,TRTAvg,TRTStd
