#!/usr/bin/env python3

"""
This comes after/during the preprocessing and builds the stage to 
split the data into four parts.
"""
import os 
import pandas as pd

def split_and_format():
    ROOT_DIR =  os.path.dirname(os.path.abspath(os.curdir))
    data_loc = os.path.join(ROOT_DIR,"data")
    save_loc = os.path.abspath(os.curdir)

    train_data = os.path.join(data_loc,"train.csv")
    val_data = os.path.join(data_loc,"dev.csv")

    train = pd.read_csv(train_data)
    t_save_FFDAvg = train[['sentence_id', 'word', 'FFDAvg']]
    t_save_FFDAvg.to_csv(os.path.join(save_loc,"Train_FFDAvg"),encoding='utf-8', index=False)
    t_save_FFDStd = train[['sentence_id', 'word', 'FFDStd']]
    t_save_FFDStd.to_csv(os.path.join(save_loc,"Train_FFDStd"),encoding='utf-8', index=False)
    t_save_TRTAvg = train[['sentence_id', 'word', 'TRTAvg']]
    t_save_TRTAvg.to_csv(os.path.join(save_loc,"Train_TRTAvg"),encoding='utf-8', index=False)
    t_save_TRTStd = train[['sentence_id', 'word', 'TRTStd']]
    t_save_TRTStd.to_csv(os.path.join(save_loc,"Train_TRTStd"),encoding='utf-8', index=False)

    val = pd.read_csv(val_data)
    v_save_FFDAvg = val[['sentence_id', 'word', 'FFDAvg']]
    v_save_FFDAvg.to_csv(os.path.join(save_loc,"Dev_FFDAvg"),encoding='utf-8', index=False)
    v_save_FFDStd = val[['sentence_id', 'word', 'FFDStd']]
    v_save_FFDStd.to_csv(os.path.join(save_loc,"Dev_FFDStd"),encoding='utf-8', index=False)
    v_save_TRTAvg = val[['sentence_id', 'word', 'TRTAvg']]
    v_save_TRTAvg.to_csv(os.path.join(save_loc,"Dev_TRTAvg"),encoding='utf-8', index=False)
    v_save_TRTStd = val[['sentence_id', 'word', 'TRTStd']]
    v_save_TRTStd.to_csv(os.path.join(save_loc,"Dev_TRTStd"),encoding='utf-8', index=False)




