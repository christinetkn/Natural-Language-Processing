import torch
import os
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable



class BaselineDNN(nn.Module):
    """
    1. We embed the words in the input texts using an embedding layer
    2. We compute the min, mean, max of the word embeddings in each sample
       and use it as the feature representation of the sequence.
    4. We project with a linear layer the representation
       to the number of classes.ngth)
    """

    def __init__(self, output_size, embeddings, trainable_emb=False):
        """
        Args:
            output_size(int): the number of classes
            embeddings(bool):  the 2D matrix with the pretrained embeddings
            trainable_emb(bool): train (finetune) or freeze the weights
                the embedding layer
        """

        super(BaselineDNN, self).__init__()

        # 1 - define the embedding layer
       
        #create embedding layer, # 2 - initialize the weights of our Embedding layer 
        #from the pretrained word embeddings
        # 3 - define if the embedding layer will be frozen or finetuned
        self.embedding_layer = nn.Embedding.from_pretrained(torch.from_numpy(embeddings),freeze = True ) # EX4
        
        
        # 4 - define a non-linear transformation of the representations
        
        self.new_layer1 = torch.nn.Linear(2*embeddings.shape[1], 128) #EX 3.1.1
        self.layer1 = torch.nn.Linear(embeddings.shape[1], 128) # EX5
        self.layer2 = nn.ReLU() # EX5
        
        # 5 - define the final Linear layer which maps
        # the representations to the classes
        self.layer3 = nn.Linear(128, output_size) # EX5
        

    def forward(self, x, lengths):
        """
        This is the heart of the model.
        This function, defines how the data passes through the network.

        Returns: the logits for each class

        """
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
        embeddings = self.embedding_layer(x) #
        
        # 2 - construct a sentence representation out of the word embeddings
        
        
        new_representations = torch.zeros([len(x), 2*embeddings.shape[2]]).to(DEVICE) #EX 3.1.1
        representations = torch.sum(embeddings, dim=1)
        
        maxE, _ = torch.max(embeddings, dim=1) #EX 3.1.1
        for i in range(len(lengths)): # EX6
            
            representations[i] = representations[i] / lengths[i] # EX6
            meanE = representations[i] / lengths[i] #EX 3.1.1
            
            new_representations[i] = torch.cat((meanE,maxE[i])) #EX 3.1.1
            
        # 3 - transform the representations to new ones.
        representations = self.layer2(self.layer1(representations))  # EX6
        new_representations = self.layer2(self.new_layer1(new_representations))
        
        # 4 - project the representations to classes using a linear layer
        logits = self.layer3(representations) # EX6
        
        logits2 = self.layer3(new_representations) #EX 3.1.1
        return logits2 #for the prelab exercise we return logits


class LSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, label_size, batch_size, embeddings): 

        super(LSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        

        self.embedding_layer = nn.Embedding.from_pretrained(torch.from_numpy(embeddings),freeze = True ) 
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        
        self.linear1 = nn.Linear(hidden_dim, label_size)
        self.linear2 = nn.Linear(3*embeddings.shape[1],label_size )


    def forward(self, x, lengths):
        
        batch_size, max_length = x.shape
        embeddings = self.embedding_layer(x)
        
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #lengths = lengths.cpu()
        
        X = torch.nn.utils.rnn.pack_padded_sequence(embeddings, lengths.cpu(), batch_first =True, enforce_sorted =False)
        ht, _ = self.lstm(X)
        ht, _ = torch.nn.utils.rnn.pad_packed_sequence(ht, batch_first =True)
        # Sentence representation as the final hidden state of the model
        
        representations = torch.zeros(batch_size, self.hidden_dim).float()
        sum_representations = torch.sum(embeddings, dim=1)
        new_representations = torch.zeros([len(x), 3*embeddings.shape[2]]).cpu()
        maxE, _ = torch.max(embeddings, dim=1)
    
        for i in range(lengths.shape[0]):
            
            last = lengths[i] - 1 if lengths[i] <= max_length else max_length - 1
            representations[i] = ht[i, last, :]
            meanE = sum_representations[i] / lengths[i]
            new_representations[i] = torch.cat((representations[i], meanE ,maxE[i] ))
        
        logits1 = self.linear1(representations) #EX3.2.1
        logits2 = self.linear2(new_representations) #EX3.2.2
        return logits2


class Bidirectional_LSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, label_size, batch_size, embeddings, bidirectional):

        super(Bidirectional_LSTM, self).__init__()
        
        self.embedding_layer = nn.Embedding.from_pretrained(torch.from_numpy(embeddings),freeze = True )
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=2, #lstm_layers
                            bidirectional=bidirectional,
                            batch_first=True)
        
        num_directions = 2 
        self.fc1 = nn.Linear(100, hidden_dim) #lstm_units = batch_size=128
        self.fc2 = nn.Linear(150, label_size)
        self.Tanh = nn.Tanh()
        #self.dropout = nn.Dropout(dropout)
        self.lstm_layers = 2
        self.num_directions = 2
        self.lstm_units = 128

    def init_hidden(self, batch_size):
        h, c = (Variable(torch.zeros(self.lstm_layers * self.num_directions, batch_size, 50)),
                Variable(torch.zeros(self.lstm_layers * self.num_directions, batch_size, 50)))
        return h, c

    def forward(self, x, lengths):
        batch_size = x.shape[0]
        
        h_0, c_0 = self.init_hidden(128)

        embedded = self.embedding_layer(x)
        packed_embedded = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted =False) #format that enables the model to ignore the padded elements.
        
        

        output, (h_n, c_n) = self.lstm(packed_embedded, (h_0, c_0))
        output_unpacked, output_lengths = pad_packed_sequence(output, batch_first=True)
        out = output_unpacked[:, -1, :]  #To get the hidden state of the last time step we used output_unpacked[:, -1, :] command and we use it to feed the next fully-connected layer      
        h = self.fc1(out)
        
        sum_representations = torch.sum(embedded, dim=1)
        new_representations = torch.zeros([len(x), 3*embedded.shape[2]]).cpu()
        
        maxE, _ = torch.max(embedded, dim=1)
        
        for i in range(lengths.shape[0]):
            
            
            meanE = sum_representations[i] / lengths[i]
            new_representations[i] = torch.cat((h[i],meanE,maxE[i]))
        
        logits1 = self.fc2(new_representations)

        
        return logits1

