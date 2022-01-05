#!/usr/bin/env python3

"""
Version:1.0
"""
import logging
import preprocess
import DataRepr
import os
import format_data
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.autograd import Variable
from transformers import get_linear_schedule_with_warmup
import random
import numpy as np
import torch
import time
import datetime

class Eye_Regression(torch.nn.Module):
    def __init__(self,input_size,out_size):
        super(Eye_Regression,self).__init__()
        self.linear = torch.nn.Linear(input_size,out_size)
    
    def forward(self,X):
        out = self.linear(X)
        return out

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_val = 13

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    logging.basicConfig()
    logging.root.setLevel(logging.NOTSET)
    logging.basicConfig(level=logging.NOTSET)
    logger = logging.getLogger("trainer")

    logger.info('Preprocessing Initiated')
    
    preprocess.preprocess()

    train_data = os.path.join(os.path.join(os.getcwd(),"data"),"train.csv")
    val_data = os.path.join(os.path.join(os.getcwd(),"data"),"dev.csv")
    
    

    # For the first experiment, we take the contextual word representation (BERT multilingual uncased) of tokens and pass it to a regression layer. 
    # We end up having 4 such models corresponding the 4 metrics to be predicted.
    T_X, T_Y1, T_Y2, T_Y3, T_Y4 = format_data.reader(train_data)    
    Test_X, Test_Y1, Test_Y2, Test_Y3, Test_Y4 = format_data.reader(val_data)

    logger.info('Obtaining representation for data')

    X,Y1,Y2,Y3,Y4 = DataRepr.make_representations(T_X, T_Y1, T_Y2, T_Y3, T_Y4,device)

    data_FFD_Avg = TensorDataset(X, Y1)
    train_size = int(0.9 * len(data_FFD_Avg))
    val_size = len(data_FFD_Avg) - train_size
    train_dataset, val_dataset = random_split(data_FFD_Avg, [train_size, val_size])
    logger.info('{:>5,} training samples'.format(train_size))
    logger.info('{:>5,} validation samples'.format(val_size))
    batch_size = 16
    train_dataloader = DataLoader(train_dataset,sampler = RandomSampler(train_dataset),batch_size = batch_size)
    validation_dataloader = DataLoader(val_dataset,sampler = SequentialSampler(val_dataset),batch_size = batch_size)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    learning_rate = 0.01
    model = Eye_Regression(768, 1)
    epochs = 100
    if torch.cuda.is_available():
        model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss() 

    total_t0 = time.time()

    for epoch in range(epochs):
        logger.info('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
        logger.info('Training...')    
        t0 = time.time()
        total_train_loss = 0
        
        model.train()

        for step, batch in enumerate(train_dataloader):
            if step % 10 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                logger.info('  Batch %d  of  %d.    Elapsed: %s.'%(step, len(train_dataloader), elapsed))
            
            w_input = batch[0].to(device)
            pred = batch[1].to(device)

            optimizer.zero_grad()
            
            outputs = model(w_input)
            
            loss = criterion(outputs, pred)

            total_train_loss += loss.item()

            # loss.backward(retain_graph=True)
            loss.backward()
            optimizer.step()

        logger.info('epoch %f, loss %f'%(epoch, loss.item()))
    
        avg_train_loss = total_train_loss / len(train_dataloader)            
    
        training_time = format_time(time.time() - t0)

        logger.info("  Average training loss: %f"%(avg_train_loss))
        logger.info("  Training epcoh took: %f"%(training_time))

        logger.info("Running Validation...")
        
        t0 = time.time()

        model.eval()
        
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        for batch in enumerate(validation_dataloader):
             
            w_input = batch[0].to(device)
            pred = batch[1].to(device)

            with torch.no_grad(): 
                outputs = model(w_input)
                loss = criterion(outputs, pred)
            
            total_eval_loss += loss.item()

        avg_val_loss = total_eval_loss / len(validation_dataloader)
        validation_time = format_time(time.time() - t0)
        logger.info("  Validation Loss: %f"%(avg_val_loss))
        logger.info("  Validation took: %f"%(validation_time))
    
    logger.info("Training complete!")
    logger.info("Total training took %s (h:mm:ss)"%(format_time(time.time()-total_t0)))
