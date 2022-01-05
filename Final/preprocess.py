#!/usr/bin/env python3
import os
import shutil
import pandas as pd
import numpy as np
# import re
# import wordfreq

"""
Preprocessing script
"""
def preprocess():
# if __name__=="__main__":
    data_location = os.path.join(os.getcwd(),"data_original")
    
    new_data_location = os.path.join(os.getcwd(),"data")
    if os.path.exists(new_data_location):
        shutil.rmtree(new_data_location)
    os.mkdir(new_data_location)
    
    # Training data section
    train_loc = os.path.join(data_location,"train.csv")
    new_train_loc = os.path.join(new_data_location,"train.csv")
    train_data = pd.read_csv(train_loc)
    sent_id = train_data["sentence_id"]
    u, indices = np.unique(sent_id, return_inverse=True)
    train_data["sentence_id"] = indices
    train_data.to_csv(path_or_buf=new_train_loc,index=False)
  
    
    # Validation data section
    dev_loc = os.path.join(data_location,"dev.csv")
    new_dev_loc = os.path.join(new_data_location,"dev.csv")
    dev_data = pd.read_csv(dev_loc)
    sent_id = dev_data["sentence_id"]
    u, indices = np.unique(sent_id, return_inverse=True)
    dev_data["sentence_id"] = indices
    dev_data.to_csv(path_or_buf=new_dev_loc,index=False)


if __name__ == '__main__':
    preprocess()