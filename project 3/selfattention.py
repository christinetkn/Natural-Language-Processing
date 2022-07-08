import torch
import os
from torch import nn
import numpy as np
from torch.autograd import Variable

class SelfAttention(nn.Module):
    def __init__(self, batch_first, non_linearity, embeddings):
        super(SelfAttention, self).__init__()

        self.batch_first = batch_first
        self.attention_weights = nn.Parameter(torch.FloatTensor(embeddings.shape[1]))
        self.softmax = nn.Softmax(dim=-1)
        
        self.hidden_dim = 50  #attention and LSTM
        self.batch_size = 128  #ATTENTION AND Lstm
        self.lstm = nn.LSTM(50, self.hidden_dim) #attention and lstm 50 is embedding dim!`~~~~!!
        
        if non_linearity == "relu":
            self.non_linearity = nn.ReLU()
        else:
            self.non_linearity = nn.Tanh()

        nn.init.uniform(self.attention_weights.data, -0.005, 0.005)
        
        
        self.embedding_layer = nn.Embedding.from_pretrained(torch.from_numpy(embeddings),freeze = True )
        self.layer3 = nn.Linear(50, 2)
    def get_mask(self, attentions, lengths):
        """
        Construct mask for padded itemsteps, based on lengths
        """
        
        max_len = max(lengths.data)
        mask = Variable(torch.ones(attentions.size())).detach()

        if attentions.data.is_cuda:
            mask = mask.cuda()

        for i, l in enumerate(lengths.data):  # skip the first sentence
            if l < max_len:
                mask[i, l:] = 0
        return mask

    def forward(self, inputs, lengths):

        ##################################################################
        # STEP 1 - perform dot product
        # of the attention vector and each hidden state
        ##################################################################

        
        batch_size, max_length = inputs.shape #lstm
        embeddings = self.embedding_layer(inputs) 

        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X = torch.nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first =True, enforce_sorted =False)
        ht, _ = self.lstm(X)
        ht, _ = torch.nn.utils.rnn.pad_packed_sequence(ht, batch_first =True)
        
        
        #scores = self.non_linearity(torch.matmul(embeddings, self.attention_weights)) #EX 3.3.1
        scores2 = self.non_linearity(torch.matmul(ht, self.attention_weights)) #EX 3.3.2
        
        #scores = self.softmax(scores) #EX3.3.1
        scores2 = self.softmax(scores2) #EX3.3.2
        ##################################################################
        # Step 2 - Masking
        ##################################################################

        # construct a mask, based on the sentence lengths
        #mask = self.get_mask(scores, lengths) #EX3.3.1
        mask2 = self.get_mask(scores2, lengths) #EX3.3.2

        # apply the mask - zero out masked timesteps
        #masked_scores = scores * mask #EX3.3.1
        masked_scores = scores2 * mask2 #EX3.3.2
        # re-normalize the masked scores
        
        _sums = masked_scores.sum(-1, keepdim=True)  # sums per row
        scores = masked_scores.div(_sums)  # divide by row sum

        ##################################################################
        # Step 3 - Weighted sum of hidden states, by the attention scores
        ##################################################################

        # multiply each hidden state with the attention weights
        #weighted = torch.mul(embeddings, scores.unsqueeze(-1).expand_as(embeddings)) #EX3.3.1
        weighted = torch.mul(ht, scores.unsqueeze(-1).expand_as(ht)) #EX3.3.2
        # sum the hidden states
        representations = weighted.sum(1).squeeze()

        return self.layer3(representations)


class BiAttentionLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, label_size, batch_size, embeddings, bidirectional, batch_first, non_linearity):
        super(BiAttentionLSTM, self).__init__()

        self.batch_first = batch_first
        self.attention_weights = nn.Parameter(torch.FloatTensor(embeddings.shape[1]))
        self.softmax = nn.Softmax(dim=-1)

        self.hidden_dim = 50  #attention and LSTM
        self.batch_size = 128  #ATTENTION AND Lstm
        

        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=2, #lstm_layers
                            bidirectional=True,
                            batch_first=True)

        if non_linearity == "relu":
            self.non_linearity = nn.ReLU()
        else:
            self.non_linearity = nn.Tanh()

        nn.init.uniform(self.attention_weights.data, -0.005, 0.005)


        self.embedding_layer = nn.Embedding.from_pretrained(torch.from_numpy(embeddings),freeze = True )
        self.layer1 = nn.Linear(100,50) #hidden_dim = 50
        self.layer3 = nn.Linear(50, 2)
        self.total = []

    def init_hidden(self, batch_size):
        h, c = (Variable(torch.zeros(2 * 2, batch_size, 50)),
                Variable(torch.zeros(2 * 2, batch_size, 50)))
        return h, c

    def get_mask(self, attentions, lengths):
        """
        Construct mask for padded itemsteps, based on lengths
        """
        max_len = max(lengths.data)
        mask = Variable(torch.ones(attentions.size())).detach()

        if attentions.data.is_cuda:
            mask = mask.cuda()

        for i, l in enumerate(lengths.data):  # skip the first sentence
            if l < max_len:
                mask[i, l:] = 0
        return mask
    
    
    def forward(self, inputs, lengths):

        ##################################################################
        # STEP 1 - perform dot product
        # of the attention vector and each hidden state
        ##################################################################

        self.inputs = inputs

        batch_size, max_length = inputs.shape #lstm
        embeddings = self.embedding_layer(inputs)

        h_0, c_0 = self.init_hidden(128)

        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X = torch.nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first =True, enforce_sorted =False)
        ht, _ = self.lstm(X)
        ht, _ = torch.nn.utils.rnn.pad_packed_sequence(ht, batch_first =True)

        
        
        ht = self.layer1(ht)
        

        scores2 = self.non_linearity(torch.matmul(ht, self.attention_weights))

        
        scores2 = self.softmax(scores2)
        ##################################################################
        # Step 2 - Masking
        ##################################################################

        # construct a mask, based on the sentence lengths
        
        mask2 = self.get_mask(scores2, lengths)

        masked_scores = scores2 * mask2
        # re-normalize the masked scores

        _sums = masked_scores.sum(-1, keepdim=True)  # sums per row
        scores = masked_scores.div(_sums)  # divide by row sum
        ##################################################################
        # Step 3 - Weighted sum of hidden states, by the attention scores
        ##################################################################

        # multiply each hidden state with the attention weights
        self.scores = scores
        weighted = torch.mul(ht, scores.unsqueeze(-1).expand_as(ht))
        
        
        # sum the hidden states
        representations = weighted.sum(1).squeeze()
        self.total.append(self.scores)       
        return self.layer3(representations)
