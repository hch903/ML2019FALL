import os
import csv
import re
import sys
import numpy as np
import pandas as pd
from numpy import genfromtxt
# import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.init as init
from gensim.models import Word2Vec
import spacy
import re
from string import punctuation as punc

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
    def __init__(self, test_dir):
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

        if test_dir != None:
            tokens = []
            with open(test_dir) as f:
            	for i, line in enumerate(f):
                    tokens.append(line.strip().split(',')[:-1])
            self.data = tokens
    
    def get_embedding(self, load=False):
        if load:
            print("--- Load word2vec")
            embed = Word2Vec.load('word2vec')
        else:
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

        return torch.LongTensor(all_indices)

    def pad_to_len(self, arr, padded_len, padding=0):
        if len(arr) > padded_len:
            return arr[:padded_len]
        else:
            for i in range(padded_len - len(arr)):
                arr.append(self.word2index["<PAD>"])
            return arr
    def print_(self):
        print(self.vectors)

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

def main():
    df = pd.read_csv(sys.argv[1])
    data = df['comment']

    nlp = spacy.load("en_core_web_sm")

    with open('testX_token_nopunc.txt', 'w') as f:   
        for i, sentence in enumerate(data):
            sentence = data_preprocess(sentence)
            for token in nlp(sentence):
                f.write('%s,' %(str(token.text)))
            f.write('\n')
    print("--- testX_token_nopunc.txt has been saved")

    preprocess = Preprocess("testX_token_nopunc.txt")


    embedding = preprocess.get_embedding(load=True)
    # data, label = preprocess.get_indices()
    test = preprocess.get_indices(test = True)

    test_dataset = hw5Dataset(data=test, train=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    input_size = len(preprocess.word2index)
    hidden_size = 250
    batch_size = 128

    model1 = RNN(embedding, input_size, hidden_size, batch_size).cuda()
    state = torch.load('./model/model_10.pth')
    model1.load_state_dict(state)
    model2 = RNN(embedding, input_size, hidden_size, batch_size).cuda()
    state = torch.load('./model/model_11.pth')
    model2.load_state_dict(state)
    model3 = RNN(embedding, input_size, hidden_size, batch_size).cuda()
    state = torch.load('./model/model_13.pth')
    model3.load_state_dict(state)

    # if use_gpu, send model / data to GPU.
    if torch.cuda.is_available():
        print("gpu!!")
        model1.cuda()
        model2.cuda()
        model3.cuda()

    model1.eval()
    model2.eval()
    model3.eval()

    # output csv
    f = open(sys.argv[2], "w")
    # f = open('pred.csv', "w")
    f.write('id,label\n')
    ans = []
    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            x = sample['data'].cuda()
            output1 = model1(x)
            output2 = model2(x)
            output3 = model3(x)
            preds = (output1+output2+output3)/3
            preds = (preds>0.5).cpu().numpy()
            for a in preds:
                ans.append(a)

    for i in range(len(ans)):
        if ans[i] == False:
            ans[i] = 0
        else:
            ans[i] = 1
        f.write(str(i)+','+str(ans[i])+'\n')
    f.close()

    # calculate similarity
    # base = pd.read_csv('pred-0.79.csv')
    # base = base['label']
    # cnt = 0
    # for i in range(len(base)):
    #     if base[i] == ans[i]:
    #         cnt += 1

    # print('Similarity to 0.79:{}'.format(cnt / len(base)))
    # print('test finish')
if __name__=="__main__":
    # parser = argparse.ArgumentParser()
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