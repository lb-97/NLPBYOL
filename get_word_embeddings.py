import gensim.downloader as api
from collections import Counter
import torch
from torch import nn
import numpy as np
from itertools import chain
import random
from nltk.corpus import wordnet
from nltk.corpus import wordnet as wn
from num2words import num2words
import pickle

POS_LIST = ['n','v','a','s','r']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('using device for embeds',device)

def get_synonyms_antonyms(word):
    
    synsets = wordnet.synsets(word)
    if not synsets:
        return None,None
    
    synonyms = set()
    antonyms = set()

    for syn in synsets:
        for l in syn.lemmas():
            synonym = l.name()
            if synonym.isalnum():
                synonyms.add(l.name().lower())
            if l.antonyms():
                antonym = l.antonyms()[0].name()
                if antonym.isalnum():
                    antonyms.add(antonym.lower())
    synonyms.discard(word)
    return list(synonyms), list(antonyms)
    
def get_pos(word):
    total = 0
    
    out = torch.tensor(np.zeros(len(POS_LIST)),dtype=torch.float).to(device)
    synsets = wordnet.synsets(word)
    if not synsets:
        return None
    pos = Counter()
    for syn in synsets:
        pos[syn.pos()]+=1
        total += 1
    for i,a_pos in enumerate(POS_LIST):
        out[i] = pos[a_pos]/total
    return out

def get_hypernyms_hyponyms(word):
    synsets = wordnet.synsets(word)
    if not synsets:
        return None,None
    
    hypernyms,hyponyms = set(),set()
    for syn in synsets:
        for hyper in syn.hypernyms():
            candidate = hyper.name().split('.')[0]
            if candidate.isalnum():
                hypernyms.add(candidate.lower())
        
        for hypo in syn.hyponyms():
            candidate = hypo.name().split('.')[0]
            if candidate.isalnum():
                hyponyms.add(candidate.lower())
                
    return list(hypernyms),list(hyponyms)

def get_word_data(word):
    # return synonyms, antonyms, hypernyms, hyponyms, pos vector
    aux_vocab = set()
    synonyms, antonyms = get_synonyms_antonyms(word)
    hypernyms,hyponyms = get_hypernyms_hyponyms(word)
    pos_vector = get_pos(word)
    
    for word in chain(synonyms or [],antonyms or [],hypernyms or [],hyponyms or []):
        aux_vocab.add(word)
    return synonyms, antonyms, hypernyms,hyponyms,pos_vector, aux_vocab

def enrich_with_aux_vocab(w2i,i2w,i2v,i2data, aux_vocab_total,vocab,wv):
    i = len(i2v)
    for w in aux_vocab_total:
        if not w in vocab:
            w2i[w] = i
            i += 1
            if w in wv:
                i2v.append(wv[w])
            else:
                i2v.append(torch.tensor(np.zeros(NUM_INPUT),dtype=torch.float))
            i2data.append(None)
    for word in aux_vocab_total:
        vocab.add(word)


def get_vector_vocab(vocab):
    
    print('loading word2vec')
    wv = api.load('word2vec-google-news-300')
    print('loading complete')
    
    w2i = dict()
    i2w = list(vocab)
    i2v = list()
    i2data = list()
    aux_vocab_total = set()
    for i,w in enumerate(i2w):
        w2i[w] = i
        if w in wv:
            i2v.append(wv[w])
            synonyms, antonyms, hypernyms,hyponyms,pos_vector, aux_vocab = get_word_data(w)
            for aux_w in aux_vocab:
                aux_vocab_total.add(aux_w)
            i2data.append((synonyms, antonyms, hypernyms,hyponyms,pos_vector))
        else:
            i2v.append([0 for _ in range(NUM_INPUT)])
            i2data.append((None,None,None,None,None))
    input_vocab = torch.tensor(i2v,dtype=torch.float)
    enrich_with_aux_vocab(w2i,i2w,i2v,i2data, aux_vocab_total,vocab,wv)
    return w2i,i2w,i2v,i2data,input_vocab

def get_vector_vocab_from_memory():
    
    with open('w2i.dill', 'rb') as file:
        w2i = pickle.load(file)
    
    with open('i2w.dill', 'rb') as file:
        i2w = pickle.load(file)

    with open('i2data.dill', 'rb') as file:
        i2data = pickle.load(file)

    embeds = torch.load('word2vec_embeds.pt')
    
    return w2i,i2w,embeds,i2data

def w2t(word,w2i,i2t):
    if not word or not word in w2i:
        return None
    return i2t[w2i[word]]

def sample_choice(group_a,group_b,w2i,i2t):
    if np.random.random()<0.5 and group_a:
        return w2t(random.choice(group_a),w2i,i2t),torch.tensor([1],dtype=torch.float).to(device)
    elif group_b:
        return w2t(random.choice(group_b),w2i,i2t),torch.tensor([0],dtype=torch.float).to(device)
    return None,None

class EmbeddingModifierNetwork(nn.Module):
    def __init__(self):
        super(EmbeddingModifierNetwork, self).__init__()
        self.hidden = nn.Linear(NUM_INPUT,NUM_HIDDEN)
        
        self.equality_layer = nn.Linear(NUM_HIDDEN,NUM_INPUT)
        self.synonym_antonym_layer = nn.Linear(NUM_HIDDEN+NUM_INPUT,1)
        self.hypernym_hyponym_layer = nn.Linear(NUM_HIDDEN+NUM_INPUT,1)
        self.pos_layer = nn.Linear(NUM_HIDDEN,len(POS_LIST))

    def forward(self,x,syn_or_ant,hyper_or_hypo):
        x = self.hidden(x)
        
        # main output layer
        main_out = self.equality_layer(x)
        main_out = torch.sigmoid(main_out)
        
        syn_ant_out = None
        if syn_or_ant is not None:
            syn_ant_out = self.synonym_antonym_layer(torch.hstack([x,syn_or_ant]))
            syn_ant_out = torch.sigmoid(syn_ant_out)
        
        hyper_hypo_out = None
        if hyper_or_hypo is not None:
            hyper_hypo_out = self.synonym_antonym_layer(torch.hstack([x,hyper_or_hypo]))
            hyper_hypo_out = torch.sigmoid(hyper_hypo_out)
            
        pos_out = self.pos_layer(x)
        pos_out = torch.softmax(pos_out,dim=0)
        
        return main_out,syn_ant_out,hyper_hypo_out,pos_out
    
NUM_HIDDEN = 512
NUM_INPUT = 300

def get_embeds(vocab,SYN_ANT_MULT,HYPER_HYPO_MULT,POS_MULT,TEST_EPOCHS=None):
    
    w2i,i2w,i2v,i2data,input_vocab = get_vector_vocab(vocab)
    i2t = torch.tensor(i2v).to(device)

    EPOCHS = int(len(i2t)*1.5)
    if TEST_EPOCHS:
        EPOCHS = TEST_EPOCHS

    model = EmbeddingModifierNetwork().to(device)
    bce_loss = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.0005)
    
    total_loss = 0
    losses = []
    for i in range(1,EPOCHS+1):
        optimizer.zero_grad()
        idx = np.random.randint(0,len(input_vocab))
        synonyms, antonyms, hypernyms,hyponyms,pos_vector = i2data[idx]

        x = i2t[idx]
        if sum(x) == 0:
            i -= 1
            continue
        # choose antonym or synonym?
        syn_or_ant,syn_out = sample_choice(synonyms,antonyms,w2i,i2t)

        # choose hyper or hypo?
        hyper_or_hypo,hyper_out = sample_choice(hypernyms,hyponyms,w2i,i2t)

        # model output
        main_out,syn_ant_out,hyper_hypo_out,pos_out = model(x,syn_or_ant,hyper_or_hypo)

        # calculate cost
        loss = torch.tensor(0,dtype=torch.float).to(device)
        for a_main_out,a_x in zip(main_out,x):
            loss += bce_loss(a_main_out,a_x)
        loss /= len(main_out)

        if pos_vector is not None:
            pos_loss = torch.tensor(0,dtype=torch.float).to(device)
            for a_pos_vector,a_pos_out in zip(pos_out,pos_vector):
                pos_loss += bce_loss(a_pos_vector,a_pos_out)
            pos_loss /= len(pos_out)
            loss += POS_MULT*pos_loss

        if syn_or_ant is not None:
            loss += SYN_ANT_MULT*bce_loss(syn_ant_out,syn_out)

        if hyper_or_hypo is not None:
            loss += HYPER_HYPO_MULT*bce_loss(hyper_hypo_out,hyper_out)

        loss.backward()
        optimizer.step()
        total_loss += loss

        losses.append(total_loss.item()/i)
        if i%1000 == 0:
            print(f'Current Avg Embedding Loss is {total_loss/i}, iter:{i}')
    
    save_name = f'S_{num2words(SYN_ANT_MULT)}_H_{num2words(HYPER_HYPO_MULT)}_P_{num2words(POS_MULT)}_E_{num2words(EPOCHS)}.pt'.replace(' ','_')
    embeds = model.hidden(i2t)
    torch.save(embeds, save_name)
    return w2i,i2w,i2v, embeds

