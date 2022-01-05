#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import re
import wordfreq

def preprocess():
    data_location = os.path.join(os.path.dirname(os.getcwd()),"data_original")
    data_plocation = os.path.join(os.path.dirname(os.getcwd()),"data")

    train_loc = os.path.join(os.path.join(data_location,"train.csv"))
    dev_loc = os.path.join(os.path.join(data_location,"dev.csv"))


    train_ploc = os.path.join(os.path.join(data_plocation,"train.csv"))
    train_data = pd.read_csv(train_loc)
    sent_id = train_data["sentence_id"]
    u, indices = np.unique(sent_id, return_inverse=True)
    train_data["sentence_id"] = indices
    train_data.to_csv(path_or_buf=train_ploc,index=False)

    dev_ploc = os.path.join(os.path.join(data_plocation,"dev.csv"))
    dev_data = pd.read_csv(dev_loc)
    sent_id = dev_data["sentence_id"]
    u, indices = np.unique(sent_id, return_inverse=True)
    dev_data["sentence_id"] = indices
    dev_data.to_csv(path_or_buf=dev_ploc,index=False)


def extract(sent):
    sent = sent.to_numpy()
    lang = sent[0,0]
    word_len = []
    log_freq = []
    words = sent[:,3]
    prev_log = []
    next_log = []
    for word in range((words.shape)[0]):
        l = len(words[word])
        word_len.append(l)
        f = wordfreq.zipf_frequency(words[word],lang)
        log_freq.append(f)
    prev_log.append(log_freq[0])
    for el in range(1,len(log_freq)):
        prev = log_freq[el-1]+log_freq[el]
        prev_log.append(prev)
    for el in range(len(log_freq)-1):
        next = log_freq[el+1]+log_freq[el]
        next_log.append(next)
    next_log.append(log_freq[-1])
    
    return words,np.array(word_len),np.array(log_freq),np.array(prev_log),np.array(next_log)

def collect(data):
    langs = np.unique(data.to_numpy()[:,0])
    Data = np.array(["word","len","log_freq","prev_log","next_log"])
    for lang in langs:
        l = data.loc[data['language'] == lang]
        sents = np.unique(l.to_numpy()[:,1])
        for sent in sents:
            f1,f2,f3,f4,f5 = extract(l.loc[l['sentence_id']==sent])
            F = (np.vstack((f1,f2,f3,f4,f5))).T
            Data = np.vstack((Data,F))
    Data = Data[1:,:]
    Predictable = data.loc[data['word'].isin(Data[:,0])].iloc[: , -5:]
    return Data[:,1:], Predictable


    
