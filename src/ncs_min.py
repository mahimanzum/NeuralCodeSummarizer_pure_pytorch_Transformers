from apex import amp
from tokenizers import ByteLevelBPETokenizer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


INPUT_DIM = 30000
OUTPUT_DIM = 24000

src_tokenizer = ByteLevelBPETokenizer()
tgt_tokenizer = ByteLevelBPETokenizer()
#                                                      train-ncs train-others
src_tokenizer.train(["../data/ncs_preprocessed_data/train-others/code.original_subtoken"], vocab_size=INPUT_DIM, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>"])


tgt_tokenizer.train(["../data/ncs_preprocessed_data/train-others/javadoc.original"], vocab_size=OUTPUT_DIM, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>"])

from torch.utils.data import Dataset, DataLoader
from joblib import Parallel, delayed
import threading
import linecache
linecache.clearcache()
import subprocess
import os
from tqdm import tqdm
import torch

MAX_SRC_LEN = 200
MAX_TGT_LEN = 50


def pad_sequences(x, max_len):
    padded = torch.ones((max_len), dtype=torch.long)
    if len(x) > max_len: padded[:] = torch.tensor(x[:max_len] , dtype=torch.long)
    else: padded[:len(x)] = torch.tensor(x, dtype=torch.long)
    return padded

class LazyDataset(Dataset):
    def __init__(self, src_tokenizer,tgt_tokenizer, src_path, tgt_path, max_len_src = MAX_SRC_LEN,max_len_tgt=MAX_TGT_LEN):
        self.src_path = src_path
        self.tgt_path = tgt_path
        self.max_len_src = max_len_src
        self.max_len_tgt = max_len_tgt
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.num_entries = sum(1 for line in open(src_path))
        self.target_entries = sum(1 for line in open(tgt_path))
        print(self.num_entries, self.target_entries)
        assert self.num_entries == self.target_entries
        
            
    def __getitem__(self, idx):
        x = self.src_tokenizer.encode("<s>"+linecache.getline(self.src_path, idx + 1).strip()+"</s>").ids
        y = self.tgt_tokenizer.encode("<s>"+linecache.getline(self.tgt_path, idx + 1).strip()+"</s>").ids
        
        return torch.tensor(pad_sequences(x,self.max_len_src), dtype=torch.long),torch.tensor(pad_sequences(y,self.max_len_tgt), dtype=torch.long) 
    
    def __len__(self):
        return self.num_entries

train_dataset = LazyDataset(src_tokenizer, tgt_tokenizer, '../data/ncs_preprocessed_data/train-others/code.original_subtoken',
                            '../data/ncs_preprocessed_data/train-others/javadoc.original')



test_dataset = LazyDataset(src_tokenizer, tgt_tokenizer, '../data/ncs_preprocessed_data/test/code.original_subtoken',
                            '../data/ncs_preprocessed_data/test/javadoc.original')



import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import spacy
import numpy as np

import random
import math
import time

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Encoder(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim,
                 dropout, 
                 device,
                 max_length = MAX_SRC_LEN):
        super().__init__()

        self.device = device
        
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim,
                                                  dropout, 
                                                  device) 
                                     for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, src, src_mask):
        
        #src = [batch size, src len]
        #src_mask = [batch size, src len]
        
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        
        #pos = [batch size, src len]
        
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        
        #src = [batch size, src len, hid dim]
        
        for layer in self.layers:
            src = layer(src, src_mask)
            
        #src = [batch size, src len, hid dim]
            
        return src

class EncoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim,  
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        
        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, src len]
                
        #self attention
        _src, _ = self.self_attention(src, src, src, src_mask)
        
        #dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        #positionwise feedforward
        _src = self.positionwise_feedforward(src)
        
        #dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        return src
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None):
        
        batch_size = query.shape[0]
        
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
                
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]
                
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]
                
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        #energy = [batch size, n heads, query len, key len]
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim = -1)
                
        #attention = [batch size, n heads, query len, key len]
                
        x = torch.matmul(self.dropout(attention), V)
        
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        
        #x = [batch size, query len, hid dim]
        
        return x, attention
    
class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        #x = [batch size, seq len, hid dim]
        
        x = self.dropout(torch.relu(self.fc_1(x)))
        
        #x = [batch size, seq len, pf dim]
        
        x = self.fc_2(x)
        
        #x = [batch size, seq len, hid dim]
        
        return x
class Decoder(nn.Module):
    def __init__(self, 
                 output_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device,
                 max_length = MAX_TGT_LEN):
        super().__init__()
        
        self.device = device
        
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim, 
                                                  dropout, 
                                                  device)
                                     for _ in range(n_layers)])
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        #trg = [batch size, trg len]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, trg len]
        #src_mask = [batch size, src len]
                
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
                            
        #pos = [batch size, trg len]
            
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
                
        #trg = [batch size, trg len, hid dim]
        
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]
        
        output = self.fc_out(trg)
        
        #output = [batch size, trg len, output dim]
            
        return output, attention
class DecoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        #trg = [batch size, trg len, hid dim]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, trg len]
        #src_mask = [batch size, src len]
        
        #self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        
        #dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
            
        #trg = [batch size, trg len, hid dim]
            
        #encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        
        #dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
                    
        #trg = [batch size, trg len, hid dim]
        
        #positionwise feedforward
        _trg = self.positionwise_feedforward(trg)
        
        #dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return trg, attention
    
class Seq2Seq(nn.Module):
    def __init__(self, 
                 encoder, 
                 decoder, 
                 src_pad_idx, 
                 trg_pad_idx, 
                 device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
    def make_src_mask(self, src):
        
        #src = [batch size, src len]
        
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        #src_mask = [batch size, 1, 1, src len]

        return src_mask
    
    def make_trg_mask(self, trg):
        
        #trg = [batch size, trg len]
        
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        
        #trg_pad_mask = [batch size, 1, 1, trg len]
        
        trg_len = trg.shape[1]
        
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        
        #trg_sub_mask = [trg len, trg len]
            
        trg_mask = trg_pad_mask & trg_sub_mask
        
        #trg_mask = [batch size, 1, trg len, trg len]
        
        return trg_mask

    def forward(self, src, trg):
        
        #src = [batch size, src len]
        #trg = [batch size, trg len]
                
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        #src_mask = [batch size, 1, 1, src len]
        #trg_mask = [batch size, 1, trg len, trg len]
        
        enc_src = self.encoder(src, src_mask)
        
        #enc_src = [batch size, src len, hid dim]
                
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        
        #output = [batch size, trg len, output dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return output, attention




HID_DIM = 512
ENC_LAYERS = 6
DEC_LAYERS = 6
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.2
DEC_DROPOUT = 0.2

enc = Encoder(INPUT_DIM, 
              HID_DIM, 
              ENC_LAYERS, 
              ENC_HEADS, 
              ENC_PF_DIM, 
              ENC_DROPOUT, 
              device)

dec = Decoder(OUTPUT_DIM, 
              HID_DIM, 
              DEC_LAYERS, 
              DEC_HEADS, 
              DEC_PF_DIM, 
              DEC_DROPOUT, 
              device)

model = Seq2Seq(enc, dec, 1, 1, device).to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)
        

model.apply(initialize_weights)
#LEARNING_RATE = 0.0005
LEARNING_RATE = 0.000005

optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index = 1)

from tqdm import tqdm
def train(model, dataset, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    for (src_, trg_) in tqdm(dataset):
        
        optimizer.zero_grad()

        src, trg = src_.to(device), trg_.to(device)
        output, _ = model(src, trg[:,:-1])

        #output = [batch size, trg len - 1, output dim]
        #trg = [batch size, trg len]
        #print(trg.shape)
        #print(output.shape)
        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim) #[:, :-1, :]
        trg = trg[:,1:].contiguous().view(-1)

        #output = [batch size * trg len - 1, output dim]
        #trg = [batch size * trg len - 1]
        #print(trg.shape)
        #print(output.shape)
        loss = criterion(output, trg)
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        #loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        epoch_loss += loss.item()
        #print(f'\tTrain Loss: {loss.item():.3f} | Train PPL: {math.exp(loss.item()):7.3f}')

        
    return epoch_loss / len(dataset)


def evaluate(model,dataset,criterion):
    
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for (src_, trg_) in tqdm(dataset):
            src, trg = src_.to(device), trg_.to(device)
            output, _ = model(src, trg[:,:-1])

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim) #[:, :-1, :]
            trg = trg[:,1:].contiguous().view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
        
    return epoch_loss / len(dataset)

def give_pred(model, tokens, device, max_len=MAX_TGT_LEN):
    model.eval()
    
    src_indexes = tokens
    #print(tokens.size())
    src_tensor = torch.LongTensor(src_indexes).to(device)
    
    src_mask = model.make_src_mask(src_tensor)
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = torch.zeros((tokens.size()[0], 1), dtype = torch.long)
    trg_indexes = trg_indexes.tolist()
    #print(trg_indexes)
    #print(trg_indexes.size())
    
    for i in range(max_len):
        #print(len(trg_indexes),len(trg_indexes[0]))
        #trg_tensor = torch.LongTensor(pad_sequences(trg_indexes, MAX_TGT_LEN)).unsqueeze(0).to(device)
        trg_tensor = torch.LongTensor(trg_indexes).to(device)
        #print("in loop", trg_tensor.size())
        trg_mask = model.make_trg_mask(trg_tensor)
        #print(trg_mask)
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        
        #print("out shape", output.size())
        
        for idx in range(len(trg_indexes)):
            #print("shape here = ", output[i][-1].size())
            #print(output[idx][-1].argmax().item())
            trg_indexes[idx].append(output[idx][-1].argmax().item())
            
        
        #pred_token = output.argmax(2)[:,-1].item()
        #print(output[0][-1].argmax())
        #print(pred_token)
        #trg_indexes.append(pred_token)

        #if pred_token == 2:
        #    break
    #print(len(trg_indexes),len(trg_indexes[0]))
    return trg_indexes

def calculate_blue(model,test_ds, device, max_len = MAX_TGT_LEN):
    model.eval()
    from google_bleu import compute_bleu
    sum_blue = 0
    data_count = 0
    for (src_, trg_) in tqdm(test_ds):
        hypothesis = give_pred(model,src_, device, max_len = MAX_TGT_LEN)
        reference = trg_.tolist()
        sum_blue +=compute_bleu([reference], hypothesis, smooth=True)[0]
        data_count+=1
        #there may be several references
        #for i in range(src_.size()[0]):
            #sum_blue += nltk.translate.bleu_score.sentence_bleu([reference[i]], hypothesis[i])
        #    data_count+=1
        #print(sum_blue/data_count)
        #print(data_count)
    return sum_blue/data_count

import warnings
import nltk
warnings.filterwarnings('ignore')

N_EPOCHS = 300
CLIP = 1

BATCH_SIZE = 160
EVAL_BATCH_SIZE = 600

train_dataset = DataLoader(train_dataset, batch_size = BATCH_SIZE, num_workers=5,
                     drop_last=True,
                     shuffle=True)

test_dataset = DataLoader(test_dataset, batch_size = EVAL_BATCH_SIZE, num_workers=5,
                     drop_last=True,
                     shuffle=False)

#added this
bleu_dataset = DataLoader(test_dataset, batch_size = 32, num_workers=5,
                     drop_last=True,
                     shuffle=False)


#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataset))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)



if __name__ == "__main__":
    
    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    best_valid_loss = float('inf')
    opt_level = 'O1'
    
    model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)

    for epoch in range(N_EPOCHS):

        start_time = time.time()
        train_loss = train(model, train_dataset, optimizer, criterion, CLIP)
        #scheduler.step()
        scheduler.step(train_loss)
        valid_loss = evaluate(model, test_dataset, criterion)

        blue_score = calculate_blue(model,bleu_dataset, device, max_len = MAX_TGT_LEN)
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)


        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'amp': amp.state_dict()
            }
            torch.save(checkpoint, '../models/best_val_no_scheduler.pt')


        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
        print("validation blue score = ",blue_score*100)
    
    
    



'''
try big lr
reduce on platue
Epoch: 94 | Time: 3m 15s
        Train Loss: 1.080 | Train PPL:   2.943
         Val. Loss: 4.529 |  Val. PPL:  92.654
validation blue score =  9.151920398731296

no scheduler
Epoch: 39 | Time: 3m 19s
        Train Loss: 2.487 | Train PPL:  12.027
         Val. Loss: 3.960 |  Val. PPL:  52.468
validation blue score =  25.177625408480335

with cosine annealing



restore
opt_level = 'O1'
model = Seq2Seq(enc, dec, 1, 1, device).to(device)
LEARNING_RATE = 0.0005

optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

checkpoint = torch.load('../models/big_checkpoint_corrected_600_loss1.pt')

model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
amp.load_state_dict(checkpoint['amp'])

'''




























