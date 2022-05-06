import gensim.downloader as api
from itertools import chain
import re
from num2words import num2words
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string, strip_punctuation,strip_non_alphanum
from torch import nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
from collections import Counter
import os
from sklearn.model_selection import train_test_split
import time 

join_path = os.path.join

t = time.time()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('using device',device)

def check_int(st):
    try:
        int(st)
        return True
    except:
        return False

    
def preprocess(s):
    s = strip_non_alphanum(s.lower())
    ans = []
    for w in s.split():
        if check_int(w):
            w = num2words(w)
            w = strip_non_alphanum(w.lower())
            for _w in w.split():
                ans.append(_w)
        else:
            ans.append(w)
    return ans

def sentence2LongTensor(sentence,w2i):
    return torch.LongTensor([w2i[w] for w in sentence]).to(device)

class TextClassificationModel(nn.Module):
    def __init__(self, embed_weights):
        super(TextClassificationModel, self).__init__()
        self.dim = embed_weights.shape[1]
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embed_weights))
        self.lstm = lstm = nn.LSTM(input_size=self.dim,
                            hidden_size=NUM_HIDDEN//2,
                            num_layers=1,
                            bidirectional=True)
        self.hidden = nn.Linear(NUM_HIDDEN,1)
        self.double()
    def forward(self, text):
        x = self.embedding(text)
        x, _ = self.lstm(x)
        x = self.hidden(x[-1])
        x = torch.sigmoid(x)
        return x
    
def gen(X,Y):
    for x,y in zip(X,Y):
        yield x,y

pos = []
neg = []

with open('pos_new.txt','r') as f:
    pos = f.readlines()
    
with open('neg_new.txt','r') as f:
    neg = f.readlines()

for i in range(len(pos)):
    pos[i] = preprocess(pos[i])
    
for i in range(len(neg)):
    neg[i] = preprocess(neg[i])
    


NUM_HIDDEN = 512
NUM_DIM = 300

vocab = set()
for s in chain(pos,neg):
    for i,w in enumerate(s):
        vocab.add(w)
        
def get_vocab():
    return vocab

def test_embeddings(w2i,embed_weights):
    pos_tensors = [sentence2LongTensor(sentence,w2i) for sentence in pos]
    neg_tensors = [sentence2LongTensor(sentence,w2i) for sentence in neg]

    stacked_tensors = neg_tensors+pos_tensors
    y = np.zeros(len(stacked_tensors))
    y[len(neg_tensors):] = 1
    y = torch.tensor(y.reshape(-1,1)).to(device)

    X_train, X_test, y_train, y_test = train_test_split(stacked_tensors, y, test_size=0.20)

    net = TextClassificationModel(embed_weights).to(device)
    loss = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.005)

    train_loss_history = []
    test_loss_history = []


    for epoch in range(5):
        train_loss = 0.0
        test_loss = 0.0
        for inp, labels in gen(X_train,y_train):
            inp = inp.to(device)
            labels = labels.to(device)
            print('.',end='')
            optimizer.zero_grad()
            predicted_output = net(inp)
            fit = loss(predicted_output,labels)

            fit.backward()
            optimizer.step()
            train_loss += fit.item()

        correct = 0
        total = 0
        for data in gen(X_test,y_test):
            with torch.no_grad():
                inp, labels = data
                predicted_output = net(inp)
                fit = loss(predicted_output,labels)
                test_loss += fit.item()
                a_out=0
                if predicted_output.item()>0.5:
                    a_out = 1
                if labels[0] == a_out:
                    correct += 1
            total += 1
        train_loss = train_loss/len(X_train)
        test_loss = test_loss/len(X_test)
        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)
        print('accuracy',correct/total)
        print('Epoch %s, Train loss %s, Test loss %s'%(epoch, train_loss, test_loss))

    return correct/total