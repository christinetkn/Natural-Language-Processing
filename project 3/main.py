import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader

from config import EMB_PATH
from dataloading import SentenceDataset
from models import BaselineDNN
from models import LSTM
from models import Bidirectional_LSTM

from selfattention import SelfAttention
from selfattention import BiAttentionLSTM


from training import train_dataset, eval_dataset
from utils.load_datasets import load_MR, load_Semeval2017A
from utils.load_embeddings import load_word_vectors
from sklearn.metrics import f1_score, accuracy_score, recall_score
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

########################################################
# Configuration
########################################################


# Download the embeddings of your choice
# for example http://nlp.stanford.edu/data/glove.6B.zip

# 1 - point to the pretrained embeddings file (must be in /embeddings folder)
EMBEDDINGS = os.path.join(EMB_PATH, "glove.6B.50d.txt")

# 2 - set the correct dimensionality of the embeddings
EMB_DIM = 50

EMB_TRAINABLE = False
BATCH_SIZE = 128
EPOCHS = 50
DATASET = "MR"  # options: "MR", "Semeval2017A"
hidden_dim = 50


# if your computer has a CUDA compatible gpu use it, otherwise use the cpu

#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")
########################################################
# Define PyTorch datasets and dataloaders
########################################################

# load word embeddings

word2idx, idx2word, embeddings = load_word_vectors(EMBEDDINGS, EMB_DIM)


# load the raw data
X_train, y_train, X_test, y_test = load_MR()


le = LabelEncoder() #EX1

######################## EX 6.6.1 ##########################################
#______________________ bow-tfidf features _________________________________


count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
keys = list(count_vect.vocabulary_.keys())

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

vec = X_train_tfidf.tocoo()
feature_names=count_vect.get_feature_names()
tuples = zip(vec.col, vec.data)

tfidf_vals = []
feature_vals = []
for idx, score in tuples:

    #keep track of feature name and its corresponding tfidf value
    tfidf_vals.append(round(score, 4)) 
    feature_vals.append(feature_names[idx])

#create a tuples of feature,score
#results = zip(feature_vals,score_vals)
results= {}
for idx in range(len(feature_vals)):
    results[feature_vals[idx]]=tfidf_vals[idx]

tfidfs = results


for word in tfidfs:
    if word in word2idx:
        embeddings[word2idx[word]] = embeddings[word2idx[word]]*tfidfs[word]
        


le.fit(y_train) #EX1
y_train = le.transform(y_train) #EX1
le.fit(y_test)#EX1
y_test = le.transform(y_test)  # EX1
n_classes = le.classes_.size  # EX1 - LabelEncoder.classes_.size

#print("The first 10 labels for MR Dataset mapped into numbers: ", y_train[:10]) #EX1

# Define our PyTorch-based Dataset
train_set = SentenceDataset(X_train, y_train, word2idx)
test_set = SentenceDataset(X_test, y_test, word2idx)


# EX4 - Define our PyTorch-based DataLoader
train_loader = torch.utils.data.DataLoader(train_set, BATCH_SIZE, shuffle=True)  # EX7 
test_loader = torch.utils.data.DataLoader(test_set, BATCH_SIZE, shuffle=False) # EX7 

#############################################################################
# Model Definition (Model, Loss Function, Optimizer)
#############################################################################

"""
######################## EX 3.1.1 ##########################################
#______________________ u=[mean(E)||max(E)] _______________________________

model1 = BaselineDNN(output_size=n_classes,  # model1 is u=[mean(E)||max(E)]
                    embeddings=embeddings,   #EX 3.1.1
                    trainable_emb=EMB_TRAINABLE)

model1.to(DEVICE)
criterion = torch.nn.BCEWithLogitsLoss()
parameters1 = []

# We optimize ONLY those parameters that are trainable (p.requires_grad==True)
for p in model1.parameters():
    if p.requires_grad:
        parameters1.append(p)

optimizer1 = torch.optim.Adam(parameters1, lr=0.0001)

######################## EX 3.2.1 ##########################################
#______________________ u=hN _______________________________________________

criterion = torch.nn.BCEWithLogitsLoss()
model2 = LSTM(EMB_DIM, hidden_dim, n_classes,BATCH_SIZE, embeddings) # model2 is u=hN
model2.to(DEVICE)

parameters2 = []

# We optimize ONLY those parameters that are trainable (p.requires_grad==True)
for p in model2.parameters():
    if p.requires_grad:
        parameters2.append(p)

optimizer2 = torch.optim.Adam(parameters2, lr=0.0001)

######################## EX 3.2.2 ##########################################
#_______________________ u= [hN||mean(E)||max(E)] __________________________

criterion = torch.nn.BCEWithLogitsLoss()
model3 = LSTM(EMB_DIM, hidden_dim, n_classes,BATCH_SIZE, embeddings) # model3 is u= [hN||mean(E)||max(E)]
model3.to(DEVICE)

parameters3 = []

# We optimize ONLY those parameters that are trainable (p.requires_grad==True)
for p in model3.parameters():
    if p.requires_grad:
        parameters3.append(p)

optimizer3 = torch.optim.Adam(parameters3, lr=0.0001)

######################## EX 3.3.1 ##########################################
#_______________________ u=sum(ai*ei) ______________________________________

criterion = torch.nn.BCEWithLogitsLoss()
model4 = SelfAttention( batch_first=False, non_linearity="tanh", embeddings=embeddings) # model4 is u=sum(ai*ei) 
#We optimize ONLY those parameters that are trainable (p.requires_grad==True)
parameters4 = []
for p in model4.parameters():
    if p.requires_grad:
        parameters4.append(p)

optimizer4 = torch.optim.Adam(parameters4, lr=0.0001)

######################## EX 3.3.2 ##########################################
#_______________________ u=sum(ai*hi) ______________________________________

criterion = torch.nn.BCEWithLogitsLoss()
model5 = SelfAttention( batch_first=False, non_linearity="tanh", embeddings=embeddings) # model5 is u=sum(ai*hi) 
#We optimize ONLY those parameters that are trainable (p.requires_grad==True)
parameters5 = []
for p in model5.parameters():
    if p.requires_grad:
        parameters5.append(p)

optimizer5 = torch.optim.Adam(parameters5, lr=0.0001)

######################## EX 3.4.1 ##########################################
#_______________________ u=bi([hN||mean(E)||max(E)]) _______________________

criterion = torch.nn.BCEWithLogitsLoss()
model6 = Bidirectional_LSTM(embedding_dim=EMB_DIM, hidden_dim = hidden_dim, label_size = n_classes, batch_size = BATCH_SIZE, embeddings = embeddings, bidirectional = True ) # model6 is u=bi([hN||mean(E)||max(E)])

parameters6 = []
for p in model6.parameters():
    if p.requires_grad:
        parameters6.append(p)

optimizer6 = torch.optim.Adam(parameters6, lr=0.0001)
"""
######################## EX 3.4.2 ##########################################
#_______________________ u=bi(sum(ai*hi))___________________________________

criterion = torch.nn.BCEWithLogitsLoss()
model7 = BiAttentionLSTM(embedding_dim=50, hidden_dim=50, label_size=n_classes, batch_size=128, embeddings=embeddings, bidirectional=True, batch_first=False, non_linearity="tanh")
parameters7 = []
for p in model7.parameters():
    if p.requires_grad:
        parameters7.append(p)

optimizer7 = torch.optim.Adam(parameters7, lr=0.0001)

#############################################################################
# Training Pipeline
#############################################################################

train_losses = []
test_losses = []

for epoch in range(1, EPOCHS + 1):
    # train the model for one epoch
    train_dataset(epoch, train_loader, model7, criterion, optimizer7)

    # evaluate the performance of the model, on both data sets
    train_loss, (y_train_gold, y_train_pred) = eval_dataset(train_loader,
                                                            model7,
                                                                criterion)
    
    train_losses.append(train_loss)
    
    test_loss, (y_test_gold, y_test_pred) = eval_dataset(test_loader,
                                                         model7,
                                                         criterion)
    
    #f.write("y_test_gold is:"+str(y_test_gold)+'\n')
    #f.write("y_test_pred is:"+str(y_test_pred)+'\n')
    test_losses.append(test_loss)

print("train accuracy", accuracy_score(y_train_gold, y_train_pred))
print("train f1 score", f1_score(y_train_gold, y_train_pred))
print("train recall", recall_score(y_train_gold, y_train_pred))    
print("test accuracy", accuracy_score(y_test_gold, y_test_pred))
print("test f1", f1_score(y_test_gold, y_test_pred))
print("test recall", recall_score(y_test_gold, y_test_pred))
#f.close()
fig = plt.figure()
plt.plot(train_losses,  label="train data")
plt.plot(test_losses,  label="test data")
fig.suptitle('BoW-Model7 u=bi(sum(ai*hi))   Loss - epochs train and test set', fontsize=10)
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Running Loss', fontsize=16)
plt.legend()
plt.show()

