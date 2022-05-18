import torch
from torch import nn
import nltk
import sys
from nltk.corpus import wordnet
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import time 
import numpy as np
import random


import benchmark_embeddings
import get_word_embeddings

NUM_HIDDEN = 512
NUM_INPUT = 300

vocab = benchmark_embeddings.get_vocab()
w2i,i2w,i2v,i2data,input_vocab,wv = get_word_embeddings.get_vector_vocab(vocab)
words = list(vocab)
for word in words:
    if word not in wv:
        words.remove(word)
        
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 

class RepNetwork(nn.Module):
    def __init__(self):
        super(RepNetwork, self).__init__()
        self.hidden = nn.Linear(NUM_INPUT,NUM_HIDDEN) 
        self.batchnorm = nn.BatchNorm1d(NUM_HIDDEN)
        self.equality_layer = nn.Linear(NUM_HIDDEN,NUM_INPUT)
        self.batchnorm2 = nn.BatchNorm1d(NUM_INPUT)
        
    def forward(self,x):
        x = self.hidden(x)
        x = self.batchnorm(x)
        x = self.equality_layer(x)
        x = self.batchnorm2(x)
        x = torch.sigmoid(x)
        return x
    
class Projection(nn.Module):
    def __init__(self):
        super(Projection, self).__init__()
        self.hidden = nn.Linear(NUM_INPUT,NUM_INPUT)
        self.batchnorm = nn.BatchNorm1d(NUM_INPUT)
        
    def forward(self,x):
        x = self.hidden(x)
#         x = self.batchnorm(x)
        return x

def get_syn_ant(word):
    synsets = wordnet.synsets(word)
    li=[]
    ant = []
    for syn in synsets:
        for l in syn.lemmas():
            synonym = l.name()
            if l.antonyms():
                antonym = l.antonyms()[0].name()
                if antonym.isalnum():
                    ant.append(antonym.lower())
    #                 print(antonym, "<--ant")
            else: 
                if synonym.isalnum():
                    li.append(synonym.lower())
    li,ant = list(set(li)),list(set(ant))
    if word in li: li.remove(word)
    if word in ant: ant.remove(word)
    return li,ant


def loss_fn(x, y):
#     print("x,y(Before): ",x.shape,y.shape)
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
#     print("x,y(After): ",x.shape,y.shape)
    return 2 - 2 * (x * y).sum(dim=-1)



def get_byol_embed(words,batch_size=5,total_batches=20346):
    words = list(vocab)
    beta = 0.99
    online = RepNetwork().to(device)
    target = RepNetwork().to(device)
    projection = Projection().to(device)
    opt = torch.optim.Adam(online.parameters(), lr=3e-4)
    EPOCHS = 10
    for epoch in range(EPOCHS):
        
        prev = 0
        tot_loss = 0
        for batch in range(total_batches):
            loss = 0 
            next = prev+batch_size
            if next>len(words):
                next = len(words) 
            input = words[prev:next]
            
            modified_input_syn = []
            syn_list_tot=[]
            modified_input_ant=[]
            ant_list_tot=[]
            flag_syn = 1
            flag_ant = 1
            L1=0
            L2=0

            for i in range(batch_size):
                word = input[i]
#                 print("Word:",word)
                syn_list, ant_list = get_syn_ant(word)
                w_vector = torch.tensor(i2v[w2i[word]])

                for syn in syn_list:
                    if syn in w2i:
#                         print("Syn:",syn)
                        syn_vector = torch.tensor(i2v[w2i[syn]])
                        modified_input_syn.append(w_vector.unsqueeze(0))
                        syn_list_tot.append(syn_vector.unsqueeze(0))

                for ant in ant_list:
                    if ant in w2i:
#                         print("Ant:", ant)
                        ant_vector = torch.tensor(i2v[w2i[ant]])
                        modified_input_ant.append(w_vector.unsqueeze(0))
                        ant_list_tot.append(ant_vector.unsqueeze(0))
            
            
            if(len(modified_input_syn)<2): flag_syn = 0
            if(len(modified_input_ant)<2): flag_ant = 0
            
            if(flag_syn==0 and flag_ant==0):
                continue
                
            if flag_syn:
                
                modified_input_syn = torch.cat(modified_input_syn,dim=0).to(device)
                syn_list_tot = torch.cat(syn_list_tot,dim=0).to(device)
                
                online_out1 = online(modified_input_syn)
                online_out1 = projection(online_out1)
                
                rev_online_out1 = online(syn_list_tot)
                rev_online_out1 = projection(rev_online_out1)
                
                with torch.no_grad():
                    target_out1 = target(syn_list_tot)
                    rev_target_out = target(modified_input_syn)
                    
                l1 = loss_fn(online_out1,target_out1)
                l2 = loss_fn(rev_online_out1,rev_target_out)
#                 print("l1,l2: ",l1.shape,l2.shape)
                L1=(l1+l2).mean()
#                 loss+=l.mean()
            
            if flag_ant:
                
                modified_input_ant = torch.cat(modified_input_ant,dim=0).to(device)
                ant_list_tot = torch.cat(ant_list_tot,dim=0).to(device)
                
                online_out2 = online(modified_input_ant)
                online_out2 = projection(online_out2)

                rev_online_out2 = online(ant_list_tot)
                rev_online_out2 = projection(rev_online_out2)

                with torch.no_grad():
                    target_out2 = target(ant_list_tot)
                    rev_target_out = target(modified_input_ant)
                l3=loss_fn(online_out2,target_out2)
                l4 = loss_fn(rev_online_out2,rev_target_out)
                L2=1-l3+1-l4
                L2=L2.mean()            

            den = 2 if(flag_ant & flag_syn) else 1
            loss+=(flag_syn*L1+flag_ant*L2)/den
#             print("mean: ",loss)
            tot_loss+=loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            for current_params, ma_params in zip(online.parameters(), target.parameters()):
                old_weight, up_weight = ma_params.data, current_params.data
                ma_params.data = old_weight * beta + (1 - beta) * up_weight
            prev=next
    
        print(f"{epoch} epoch loss:{tot_loss/total_batches}")
        save_path_online = "./online_{}".format(epoch)
        torch.save({
            'model_state_dict': online.state_dict(),
            'optimizer_state_dict': opt.state_dict()},save_path_online)

class TextClassificationModel(nn.Module):
    def __init__(self, embed_weights,enhance=True):
        super(TextClassificationModel, self).__init__()
        self.dim = embed_weights.shape[1]
        self.embed_weights = embed_weights
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embed_weights))
        self.lstm = lstm = nn.LSTM(input_size=self.dim,
                            hidden_size=NUM_HIDDEN//2,
                            num_layers=1,
                            bidirectional=True,batch_first=False)
        self.hidden = nn.Linear(NUM_HIDDEN,1)
        self.enhance = enhance
        self.double()
    def forward(self, text):
        x=torch.tensor(0)
        if(self.enhance):
            for i in range(text.shape[0]):
                if(i==0):
                    x=(self.embed_weights[text[i]]).to(torch.float64)
                else:
                    x=torch.vstack((x,self.embed_weights[text[i]]))
        else:
            
            for i in range(text.shape[0]):
                if(i==0):
                    x=torch.tensor(i2v[text[i]]).to(float)
                else:
                    x=torch.vstack((x,torch.tensor(i2v[text[i]])))
        x = x.unsqueeze(1).to(device)
        x, _ = self.lstm(x)
        x = self.hidden(x[-1])
        x = torch.sigmoid(x)
        return x

    
def gen(X,Y):
    temp = list(zip(X, Y))
    random.shuffle(temp)
    X, Y = zip(*temp)
    for x,y in zip(X,Y):
        yield x,y
        

def test_embeddings(enhance=True):
    model = RepNetwork()
    # model.load_state_dict(torch.load("/scratch/vb2183/DL/MiniProject/trails2/online_8")['model_state_dict'])
    model.load_state_dict(torch.load("./online_8.pt")['model_state_dict'])
    print("Loading online_8.pt")
    with torch.no_grad():
        embeds_b = model.hidden(torch.tensor(i2v).to('cpu'))
    embed_weights = embeds_b.clone().detach()
    if not enhance:
        embed_weights = torch.zeros((1,300))
    
    pos_tensors = [benchmark_embeddings.sentence2LongTensor(sentence,w2i) for sentence in benchmark_embeddings.pos]
    neg_tensors = [benchmark_embeddings.sentence2LongTensor(sentence,w2i) for sentence in benchmark_embeddings.neg]
    stacked_tensors = neg_tensors+pos_tensors
    y = np.zeros(len(stacked_tensors),dtype='int')
    y[len(neg_tensors):] = 1
    y = torch.tensor(y).reshape(-1,1).to(device)
    X_train, X_test, y_train, y_test = train_test_split(stacked_tensors, y,test_size=0.20,stratify=list(np.digitize(list(y.cpu()),[0,1])))
    net = TextClassificationModel(embed_weights,enhance).to(device)
    loss = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.005)

    train_loss_history = []
    test_loss_history = []


    for epoch in range(10):
        train_loss = 0.0
        test_loss = 0.0
        track=0
        for inp, labels in gen(X_train,y_train):
            inp = inp.to(device)
            labels = labels.to(device)
#             print(f"{track} ",end='')
            optimizer.zero_grad()
            predicted_output = net(inp)
            fit = loss(predicted_output[0].float(),labels.float())
            fit.backward()
            optimizer.step()
            train_loss += fit.item()
            track+=1

        correct = 0
        total = 0
        for data in gen(X_test,y_test):
            with torch.no_grad():
                inp, labels = data
                inp = inp.to(device)
                labels = labels.to(device)
                predicted_output = net(inp)
                fit = loss(predicted_output[0].float(),labels.float())
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
#         print(f'accuracy: {correct/total}')
        print('accuracy: {}'.format(correct/total))
#         print(f'Epoch {epoch}: Train loss {train_loss}, Test loss {test_loss}')
        print('Epoch {}: Train loss {}, Test loss {}'.format(epoch,train_loss,test_loss))

    return correct/total

def pca(word='get'):
    model = RepNetwork()
    model.load_state_dict(torch.load("./online_8",map_location=device)['model_state_dict'])
    with torch.no_grad():
        embeds = model.hidden(torch.tensor(i2v).to('cpu'))
    embedX = embeds.clone().detach()
    syns,ants = get_syn_ant(word)
    s_en = embedX[w2i[word]]
    s = torch.tensor(i2v[w2i[word]])
    for syn in syns:
        if syn in w2i:
            s_en = torch.vstack((s_en,embedX[w2i[syn]]))
            s = torch.vstack((s,torch.tensor(i2v[w2i[syn]])))
    _,_,v1 = torch.pca_lowrank(s_en)
    _,_,v2 = torch.pca_lowrank(s)

    x1 = torch.matmul(s_en,v1[:,:2])
    x2 = torch.matmul(s,v2[:,:2])

    plt.figure()
    plt.scatter(x1[:,0],x1[:,1],label="Enhanced Embeddings")
    plt.scatter(x2[:,0],x2[:,1],label="Word2Vec Embeddings")
    plt.title(f"Plotted embedding of synonymys of word '{word}' after PCA reduction to 2 dimensions")
    plt.legend()
    plt.xlabel('1st dim')
    plt.ylabel('2nd dim')
    plt.show()



# get_byol_embed(list(vocab),5,20346) 


# test_embeddings(w2i,torch.zeros((1,300)),enhance=False)
# test_embeddings(w2i,embeds)

