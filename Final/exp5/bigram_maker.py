import os 
import numpy as np
import pandas as pd

def make_bigrams():
    root = os.getcwd()
    Files = ["Train_FFDAvg","Train_FFDStd","Train_TRTAvg","Train_TRTStd","Dev_FFDAvg","Dev_FFDStd","Dev_TRTAvg","Dev_TRTStd"]
    for f in Files:
        D = np.array(["sentence_id","word","metric"])
        f_ = os.path.join(root,f)
        content = pd.read_csv(f_)
        c_content = pd.read_csv(f_)
        sents = np.unique(content["sentence_id"].to_numpy())
        for sent in sents:
            sent_block = (content.loc[content['sentence_id'] == sent]).to_numpy()
            sent_block2 = (c_content.loc[c_content['sentence_id'] == sent]).to_numpy()
            for i in range(sent_block.shape[0]):
                if i==0:
                    D = np.vstack((D,sent_block2[i]))
                else:    
                    bigram = sent_block[i-1][1] + " " + sent_block[i][1]
                    sent_block2[i][1]= bigram
                    D = np.vstack((D,sent_block2[i]))
        col = (f.split("_"))[1]
        ar = pd.DataFrame(D[1:,:], columns = ['sentence_id','word',col])
        name_ = os.path.join(root,f.lower())
        ar.to_csv(name_,index=False)
