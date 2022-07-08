import torch
import os
from torch import nn
import numpy as np
from torch.autograd import Variable
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from config import EMB_PATH
from dataloading import SentenceDataset
from training import train_dataset, eval_dataset
from utils.load_datasets import load_MR, load_Semeval2017A
from utils.load_embeddings import load_word_vectors
from selfattention import BiAttentionLSTM
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, recall_score

EMBEDDINGS = os.path.join(EMB_PATH, "glove.6B.50d.txt")
EMB_DIM = 50
EMB_TRAINABLE = False
BATCH_SIZE = 128
EPOCHS = 50
DATASET = "MR"  # options: "MR", "Semeval2017A"
hidden_dim = 50

#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

word2idx, idx2word, embeddings = load_word_vectors(EMBEDDINGS, EMB_DIM)
X_train, y_train, X_test, y_test = load_MR()

X = X_test
attention_weights = nn.Parameter(torch.FloatTensor(embeddings.shape[1])) #collect the attention weights
nn.init.uniform(attention_weights.data, -0.005, 0.005) #load the input Tensor with values drawn from the uniform distribution U(-0.005, 0.005)


le = LabelEncoder() 

le.fit(y_train) 
y_train = le.transform(y_train) 
le.fit(y_test)
y_test = le.transform(y_test)  
n_classes = le.classes_.size  

train_set = SentenceDataset(X_train, y_train, word2idx)
test_set = SentenceDataset(X_test, y_test, word2idx)

train_loader = torch.utils.data.DataLoader(train_set, BATCH_SIZE, shuffle=True)  
test_loader = torch.utils.data.DataLoader(test_set, BATCH_SIZE, shuffle=False) 

criterion = torch.nn.BCEWithLogitsLoss()
model5 = BiAttentionLSTM(embedding_dim=50, hidden_dim=50, label_size=2, batch_size=128, embeddings=embeddings, bidirectional=True, batch_first=False, non_linearity="tanh")
model5.to(DEVICE)
parameters5 = []

for p in model5.parameters():
    if p.requires_grad:
        parameters5.append(p)
optimizer5 = torch.optim.Adam(parameters5, lr=0.0001)

label = []
prediction = []
attention = []
ids = [i for i in range(len(X_test))]

def save_checkpoint(state, is_best, filename='checkpoint'):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print ("=> Saving a new best")
        torch.save(state, filename)  # save checkpoint
    else:
        print ("=> Validation Accuracy did not improve")
best_accuracy = 0

f = open("validation_preds.txt", "w")
for epoch in range(1, EPOCHS + 1):
    # train the model for one epoch
    train_dataset(epoch, train_loader, model5, criterion, optimizer5)
    """
    # evaluate the performance of the model, on both data sets
    train_loss, (y_train_gold, y_train_pred) = eval_dataset(train_loader,
                                                            model5,
                                                                criterion)
    """
    
    test_loss, (y_test_gold, y_test_pred) = eval_dataset(test_loader,
                                                         model5,
                                                         criterion)                                                       
    f.write("y_test_pred is:"+str(y_test_pred)+'\n')
    label.append(y_test_gold)
    prediction.append(y_test_pred)
    attention.append(model5.scores)

    #test_losses.append(test_loss)

    # Get bool not ByteTensor
    is_best = bool(accuracy_score(y_test_gold[:epoch], y_test_pred[:epoch]) > best_accuracy)
    # Get greater Tensor to keep track best acc
    if is_best:
        best_accuracy = accuracy_score(y_test_gold[:epoch], y_test_pred[:epoch])
        best_epoch = epoch
        best_is_best = is_best
# Save checkpoint if is a new best
save_checkpoint({
    'epoch': best_epoch + 1,
    'state_dict': model5.state_dict(),
    'best_accuracy': best_accuracy
    }, best_is_best)
    
f.close()


cuda = torch.cuda.is_available()
if cuda:
    checkpoint = torch.load('checkpoint')
else:
    # Load GPU model on CPU
    checkpoint = torch.load('checkpoint')
start_epoch = checkpoint['epoch']
best_accuracy = checkpoint['best_accuracy']
model5.load_state_dict(checkpoint['state_dict'])



f = open("data.json", "w")
f.write("[")



L = model5.total
R = L[::-1]
R = R[:6]
R = R[::-1]


for i in range(len(X)-1):
    x = i // 128
    y = i % 128
    f.write('{"text": [')
    for b in range(len(X[y])-1):
        if X[y][b] != '"':
            f.write('"'+X[y][b]+'"'+',')
        else:
            continue
    f.write('"'+X[y][b+1]+'"'+'],')
    f.write('"label": ' + str(label[x][y]) + ',')
    f.write('"prediction": ' + str(prediction[x][y]) + ',')
    f.write('"attention": [')
    for a in range(len(R[x][y])-1): #this should be changed
        f.write(str((R[x][y][a]).item())+',')
    f.write(str((R[x][y][a+1]).item())+ '],')
    f.write('"id": "sample_'+ str(i)+'"},')

x = (i+1) // 128
y = (i+1) % 128

f.write('{"text": [')
for b in range(len(X[y])-1):
    if X[y][b] != '"' :
        f.write('"'+X[y][b]+'"'+',')
    else:
        continue
f.write('"'+X[y][b+1]+'"'+'],')

f.write('"label": ' + str(label[x][y]) + ',')
f.write('"prediction": ' + str(prediction[x][y]) + ',')
f.write('"attention": [')
for a in range(len(R[x][y])-1): 
    f.write(str((R[x][y][a]).item())+',')
f.write(str((R[x][y][a+1]).item())+ '],')
f.write('"id": "sample_'+ str(i)+'"}]')

