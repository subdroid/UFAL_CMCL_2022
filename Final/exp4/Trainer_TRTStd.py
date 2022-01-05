#!/usr/bin/env python3

"""
Experiment 1
"""
import make_data
import logging
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
import torch
from torch import nn
import os
import pandas as pd
from torch.utils.data import DataLoader, random_split, RandomSampler, SequentialSampler
import time
import datetime
import random
import numpy as np
from sklearn.metrics import mean_absolute_error
from bigram_maker import make_bigrams

seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

class loader_module(torch.utils.data.Dataset):
    def __init__(self, word_tokens, pred): 
        self.word_tokens = word_tokens
        self.to_predict = pred

    def __getitem__(self, id):
        item = {}
        item['input'] = torch.tensor(self.word_tokens[id],dtype=torch.long)
        item['labels'] = torch.tensor(self.to_predict[id],dtype=torch.float)
        return item
    
    def __len__(self):
        return len(self.to_predict)

def model_selector(model_name):
    if model_name=='bert':
            model = "bert-base-multilingual-uncased"
    elif model_name=='roberta':
            model = 'xlm-roberta-base'
    tokenizer = AutoTokenizer.from_pretrained(model)  
    model_out = AutoModel.from_pretrained(model, output_hidden_states=False)
    return tokenizer,model_out
    
class model(nn.Module):
    def __init__(self,base):
        super(model, self).__init__() 
        self.base = base
        self.flatten = nn.Flatten() 
        self.Linear = nn.Sequential(nn.Linear(153600, 768),nn.ReLU())
        self.linear1 = nn.Sequential(nn.Linear(768, 512),nn.ReLU())
        self.linear2 = nn.Sequential(nn.Linear(512, 32),nn.ReLU())
        self.predict = nn.Sequential(nn.Linear(32, 1))   
             
        

    def forward(self, x):
        bert_out = self.base(x)[0] # shape=[batch_size,padded_words,768]
        flat = self.flatten(bert_out)
        L = self.Linear(flat)
        l1 = self.linear1(L)
        l2 = self.linear2(l1)         
        predict = torch.squeeze(self.predict(l2), 1)
        return predict

def load_csv(cat):
    loc = os.getcwd()
    train = pd.read_csv(os.path.join(loc,"train_"+cat))
    dev = pd.read_csv(os.path.join(loc,"dev_"+cat))
    return train,dev

def time_passed(elapsed):
    elapsed = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed))

def accuracy(preds, labels):
    return mean_absolute_error(labels, preds)

if __name__ == '__main__':
    logging.basicConfig()
    logging.root.setLevel(logging.NOTSET)
    logging.basicConfig(level=logging.NOTSET)
    log = logging.getLogger("training log")
    log.info("Initiating data copy")
    make_data.split_and_format()
    make_bigrams()
    log.info("Category: TRTStd")
    log.info("Loading BERT tokenizer and model")
    B_tok, B_model = model_selector("bert")
    log.info("Selecting device")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    b_model = model(B_model)
    b_model.to(device)
    log.info("Initiate tokenization")
    train,dev = load_csv("trtstd")
    train_tokens= B_tok.batch_encode_plus(train['word'].tolist(),max_length=200,padding='max_length')['input_ids']
    train_dataset = loader_module(train_tokens, train['TRTStd'])
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    dev_tokens= B_tok.batch_encode_plus(dev['word'].tolist(),max_length=200,padding='max_length')['input_ids']
    dev_dataset = loader_module(dev_tokens, dev['TRTStd'])
    dev_loader = DataLoader(dev_dataset, batch_size=8)
    
  
    max_epochs = 100

    SAVE_PATH = os.path.join(os.getcwd(),os.path.join("saved_models","bigram_TRTStd_100.pt"))
    loss_func = torch.nn.MSELoss()  
    optimizer = AdamW(b_model.parameters(),lr = 2e-5,eps = 1e-8)
    total_steps = len(train_loader) * max_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps = 0,num_training_steps = total_steps)
    
    best_acc = 10000

    for epoch in range(max_epochs):
        log.info("epoch %d of %d"%(epoch+1,max_epochs))
        t0 = time.time()
        total_train_loss = 0
        b_model.train()
        for step, batch in enumerate(train_loader):
            elapsed = time_passed(time.time() - t0)
            # if step%20==0 and step!=0:
            #     log.info("batch %d of %d. Elapsed:%s"%(step,len(train_loader),elapsed))
            
            input_ids = batch['input'].to(device)
            labels = batch['labels'].to(device)
    
            b_model.zero_grad()  
            
            
            outputs = b_model(input_ids)
            loss = loss_func(outputs, labels)
            
            # log.info("loss %f"%loss.item())
            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(b_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        avg_train_loss = total_train_loss / len(train_loader)            
        training_time = time_passed(time.time() - t0)
        log.info("Average training loss: %f"%(avg_train_loss))
        log.info("Training epoch duration: %s"%(training_time))
        
        log.info("Validation phase")
        b_model.eval()
        total_eval_loss = 0
        total_eval_accuracy = 0

        t0 = time.time()
        for step, batch in enumerate(dev_loader):
            input_ids = batch['input'].to(device)
            labels = batch['labels'].to(device)
            outputs = b_model(input_ids)
            loss = loss_func(outputs, labels)
            total_eval_loss += loss.item()
            labels = labels.detach().cpu().numpy()
            outputs = outputs.detach().cpu().numpy()
            acc = accuracy(labels,outputs)
            total_eval_accuracy += acc



        avg_val_loss = total_eval_loss / len(dev_loader)
        avg_accuracy = total_eval_accuracy / len(dev_loader)
        validation_time = time_passed(time.time() - t0)
        log.info("  Validation loss: %f"%(avg_val_loss))
        log.info("  Validation duration: %s"%(validation_time))
        log.info("  Validation accuracy: %s"%(avg_accuracy))
        if avg_accuracy<best_acc:
            best_acc = avg_accuracy
            torch.save({'epoch': epoch,
            'model_state_dict': b_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, SAVE_PATH)
        


        


