from transformers import AutoTokenizer, AutoModel
import torch

class Model_sent:
    def __init__(self, model):
        """Decide upon the choice of models later"""
        if model=='bert':
            self.model = "bert-base-multilingual-uncased"
        elif model=='roberta':
            self.model = 'xlm-roberta-base'
     
    def init_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)  
        self.model_out = AutoModel.from_pretrained(self.model, output_hidden_states=False)
        return self.tokenizer,self.model_out
    
    def tok(self,sent_seq):
        Tok = []
        for word in sent_seq:
            encoded = self.tokenizer.encode(text=word,return_tensors = 'pt')
            Tok.append(encoded)
        return Tok
    
    def word_repr(self,tokens):
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # out_model = self.model_out.to(device)
        outputs = self.model_out(tokens)
        # outputs = out_model(tokens)
        out_words = outputs[0] # shape=[1,w,768]...final output
        # pooled_output = outputs[1] #shape=[1,768] --> hidden state corresponding to the first token
        return torch.squeeze(out_words)
    
    

    def get_representations(self,sent_seq):
        tok = self.tok(sent_seq)
        Sent = []
        for word in tok:
            word_rep = self.word_repr(word)
            """SUM OF WORD REPRESENTATIONS"""
            word_rep = torch.sum(word_rep, dim=0)
            Sent.append(word_rep)
        Rep = torch.stack([s for s in Sent])
        return Rep
        

def make_representations(T_X, T_Y1, T_Y2, T_Y3, T_Y4,device):    
    model_bert = Model_sent('bert')
    model_bert.init_model()
    Ids = []
    FFD_Avg = []
    FFD_Std = []
    TRT_Avg = []
    TRT_Std = []

    for sent,w1,w2,w3,w4 in zip(T_X,T_Y1,T_Y2,T_Y3,T_Y4):
        representations = model_bert.get_representations(sent)
        Ids.append(representations)
        FFD_Avg.append(torch.FloatTensor(w1))
        FFD_Std.append(torch.FloatTensor(w2))
        TRT_Avg.append(torch.FloatTensor(w3))
        TRT_Std.append(torch.FloatTensor(w4))
        
    
    final_representations = Ids[0]
    final_FFD_Avg = FFD_Avg[0]
    final_FFD_Std = FFD_Std[0]
    final_TRT_Avg = TRT_Avg[0]
    final_TRT_Std = TRT_Std[0]
    
    for idi in range(1,len(Ids)):
        final_representations = torch.cat((final_representations,Ids[idi]))
        final_FFD_Avg = torch.cat((final_FFD_Avg,FFD_Avg[idi]))
        final_FFD_Std = torch.cat((final_FFD_Std,FFD_Std[idi]))
        final_TRT_Avg = torch.cat((final_TRT_Avg,TRT_Avg[idi]))
        final_TRT_Std = torch.cat((final_TRT_Std,TRT_Std[idi]))

    return final_representations,torch.unsqueeze(final_FFD_Avg, 1),torch.unsqueeze(final_FFD_Std, 1),torch.unsqueeze(final_TRT_Avg, 1),torch.unsqueeze(final_TRT_Std, 1)
    