import os
import csv
import sys
import re
import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.init as init
import spacy
from gensim.models import Word2Vec
from numpy import genfromtxt
import re
from string import punctuation as punc

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)

def data_preprocess(text, use_stem = True):
    string = ''.join([c for c in text if c not in punc])
    string = string.lower()
    string = string.replace("i ' m", "im").replace("you ' re","youre").replace("didn ' t","didnt")    .replace("can ' t","cant").replace("haven ' t", "havent").replace("won ' t", "wont")    .replace("isn ' t","isnt").replace("don ' t", "dont").replace("doesn ' t", "doesnt")    .replace("aren ' t", "arent").replace("weren ' t", "werent").replace("wouldn ' t","wouldnt")    .replace("ain ' t","aint").replace("shouldn ' t","shouldnt").replace("wasn ' t","wasnt")    .replace(" ' s","s").replace("wudn ' t","wouldnt").replace(" .. "," ... ")    .replace("couldn ' t","couldnt")
    for same_char in re.findall(r'((\w)\2{2,})', string):
        string = string.replace(same_char[0], same_char[1])
    for digit in re.findall(r'\d+', string):
        string = string.replace(digit, "1")
    for punct in re.findall(r'([-/\\\\()!"+,&?\'.]{2,})',string):
        if punct[0:2] =="..":
            string = string.replace(punct, "...")
        else:
            string = string.replace(punct, punct[0])
    return string

class Preprocess(): 
    def __init__(self, data_dir, label_dir):
        self.embed_dim = 300
        self.seq_len = 40
        self.wndw_size = 3
        self.word_cnt = 3
        self.save_name = 'word2vec'
        self.index2word = []
        self.word2index = {}
        self.vectors = []
        self.unk = "<UNK>"
        self.pad = "<PAD>"
        if data_dir != None:
            tokens = []
            with open(data_dir) as f:
            	for i, line in enumerate(f):
                    tokens.append(line.strip().split(',')[:-1])
            self.data = tokens
            
        if label_dir!=None:
            dm = pd.read_csv(label_dir)
            self.label = [int(i) for i in dm['label']]
    
    def get_embedding(self, load=False):
        print("--- Get embedding")
        embed = Word2Vec(self.data, size=self.embed_dim, window=self.wndw_size, min_count=self.word_cnt, iter=16, workers=8)
        embed.save(self.save_name)
        for i, word in enumerate(embed.wv.vocab):
            # print('--- get words #{}'.format(i+1), end='\r')
            self.word2index[word] = len(self.word2index)
            self.index2word.append(word)
            self.vectors.append(embed[word])
        self.vectors = torch.tensor(self.vectors)
        self.add_embedding(self.pad)
        self.add_embedding(self.unk)
        print("--- total words: {}".format(len(self.vectors)))
        return self.vectors

    def add_embedding(self, word):
        vector = torch.empty(1, self.embed_dim)
        torch.nn.init.uniform_(vector)
        self.word2index[word] = len(self.word2index)
        self.index2word.append(word)
        self.vectors = torch.cat([self.vectors, vector], 0)
    
    def get_indices(self,test=False):
        if test == False:
            data = self.data
        all_indices = []
        for i, sentence in enumerate(data):
            # print('--- sentence count #{}'.format(i+1), end='\r')
            sentence_indices = []
            for word in sentence:
                if word in self.word2index.keys():
                    sentence_indices.append(self.word2index[word])
                else:
                    sentence_indices.append(self.word2index["<UNK>"])
                """
                if self.index2word == word and word != "<UNK>":
                    sentence_indices.append(self.word2index[word])
                else:
                    sentence_indices.append(self.word2index["<UNK>"])
                """
            sentence_indices = self.pad_to_len(sentence_indices, self.seq_len, self.word2index[self.pad])
            all_indices.append(sentence_indices)
        if test:
            return torch.LongTensor(all_indices)         
        else:    
            return torch.LongTensor(all_indices), torch.LongTensor(self.label)

    def pad_to_len(self, arr, padded_len, padding=0):
        if len(arr) > padded_len:
            return arr[:padded_len]
        else:
            for i in range(padded_len - len(arr)):
                arr.append(self.word2index["<PAD>"])
            return arr
    def print_(self):
        print(self.vectors)

# Dataset
class hw5Dataset(Dataset):
    def __init__(self, data, train=True, label=None):
        self.data = torch.LongTensor(data)
        if train:
            self.label = torch.from_numpy(label).float()
        self.train = train

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if self.train:
            sample = {'data': self.data[idx],
                      'label': self.label[idx]}
        else:
            sample = {'data': self.data[idx]}

        return sample

#model
class RNN(nn.Module):
    def __init__(self, embedding, input_size, hidden_size, batch_size, num_layers=1, fix_emb=True):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = 2

        self.embedding = nn.Embedding(input_size, 300)
        self.embedding.weight = torch.nn.Parameter(embedding)
        self.embedding.weight.requires_grad = True
        # if fix_emb else True

        self.gru = nn.GRU(300, hidden_size,
                           num_layers=self.num_layers,
                           bidirectional=True, batch_first=True, dropout=0.5)

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_size * 1*2, 128),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5)
        )

        self.fc3 = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        embed_input = self.embedding(input.long())
        output, (hidden) = self.gru(embed_input)
        a,b,c,d = hidden[0], hidden[1], hidden[2], hidden[3]
        hidden = torch.cat((c,d),1)
    
        label = self.fc3(self.fc1(hidden))
        return label.squeeze()


def main():
    df = pd.read_csv(sys.argv[1])
    data = df['comment']

    nlp = spacy.load("en_core_web_sm")
    with open('trainX_token_nopunc.txt', 'w') as f:
        for i, sentence in enumerate(data):
            sentence = data_preprocess(sentence)
            
            for token in nlp(sentence):
                f.write('%s,' %(str(token.text)))
            f.write('\n')
    print('trainX_token_nopunc.txt')

    preprocess = Preprocess("trainX_token_nopunc.txt", sys.argv[2])

    embedding = preprocess.get_embedding(load=False)
    data, label = preprocess.get_indices()

    label = pd.read_csv(sys.argv[2])
    label = label.iloc[:,1]
    label = np.asarray(label)

    # split train data
    train_vec, valid_vec = data[0:12000], data[12000:]
    train_label, valid_label = label[0:12000], label[12000:]

    # parameter
    input_size = len(preprocess.word2index)
    batch_size = 32
    n_epochs = 30
    print_every = 1
    hidden_size = 250
    lr = 0.0001

    train_dataset = hw5Dataset(data=train_vec, label=train_label, train=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = hw5Dataset(data=valid_vec, label=valid_label, train=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)


    # train
    model = RNN(embedding, input_size, hidden_size, batch_size).cuda()
    model.apply(weight_init)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    criterion = nn.BCELoss()

    model.train()

    ltrain_loss = []
    lvalid_loss = []
    ltrain_acc = []
    lvalid_acc = []
    print('--- start training')
    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        for i, sample in enumerate(train_loader):
            x = sample['data'].cuda()
            label = sample['label'].cuda()
            
            optimizer.zero_grad()
            #print(x[0])
            output_label = model(x)
            loss = criterion(output_label, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss
            #_, preds = torch.max(output_label.data, 1)
            preds = (output_label>0.5).float()
            epoch_acc += torch.sum(preds == label)

        if epoch % print_every == 0:
            model.eval()
            with torch.no_grad():
                valid_acc = 0
                valid_loss = 0
                for i, sample in enumerate(valid_loader):
                    x = sample['data'].cuda()
                    label = sample['label'].cuda()
                    optimizer.zero_grad()
                    output_label = model(x)
                    loss = criterion(output_label, label)
                    #_, preds = torch.max(output_label.data, 1)
                    valid_loss += criterion(output_label, label)
                    preds = (output_label>0.5).float()
                    valid_acc += torch.sum(preds == label)

            print('[ (%d %d%%), Loss:  %.3f, train_Acc: %.5f, valid_Loss: %.3f, valid_Acc: %.5f]' %
                (epoch,
                epoch / n_epochs * 100,
                epoch_loss/len(train_loader),
                float(epoch_acc) / len(train_loader) / batch_size,
                valid_loss/len(valid_loader),
                float(valid_acc) / len(valid_loader) / batch_size))
            ltrain_loss.append(epoch_loss/len(train_loader))
            lvalid_loss.append(valid_loss/len(valid_loader))
            ltrain_acc.append(float(epoch_acc) / len(train_loader) / batch_size)
            lvalid_acc.append(float(valid_acc) / len(valid_loader) / batch_size)
            epoch_loss = epoch_acc = 0
            if valid_loss/len(valid_loader) < 0.47:
                break
        torch.save(model.state_dict(), "./model/model_{}.pth".format(epoch))
    print('--- training finished')

        
if __name__=="__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('model_dir', type=str, help='[Output] Your model checkpoint directory')
    # parser.add_argument('--lr', default=0.001,type=float)
    # parser.add_argument('--batch', default=128, type=int)
    # parser.add_argument('--epoch', default=10, type=int)
    # parser.add_argument('--num_layers', default=1, type=int)
    # parser.add_argument('--seq_len', default=40, type=int)
    # parser.add_argument('--word_dim', default=300, type=int)
    # parser.add_argument('--hidden_dim', default=300, type=int)
    # parser.add_argument('--wndw', default=3, type=int)
    # parser.add_argument('--cnt', default=3, type=int)
    # args = parser.parse_args()
    main()

    
