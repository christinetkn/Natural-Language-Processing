from torch.utils.data import Dataset
import os
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
class SentenceDataset(Dataset):
    """
    Our custom PyTorch Dataset, for preparing strings of text (sentences)
    What we have to do is to implement the 2 abstract methods:

        - __len__(self): in order to let the DataLoader know the size
            of our dataset and to perform batching, shuffling and so on...

        - __getitem__(self, index): we have to return the properly
            processed data-item from our dataset with a given index
    """

    def __init__(self, X, y, word2idx):
        """
        In the initialization of the dataset we will have to assign the
        input values to the corresponding class attributes
        and preprocess the text samples

        -Store all meaningful arguments to the constructor here for debugging
         and for usage in other methods
        -Do most of the heavy-lifting like preprocessing the dataset here


        Args:
            X (list): List of training samples
            y (list): List of training labels
            word2idx (dict): a dictionary which maps words to indexes
        """

        for i in range(len(X)): #EX2
            

            X[i] = X[i].split(" ") #EX2
        
        self.data = X #EX2
        self.labels = y #EX2
        self.word2idx = word2idx #EX2
        print('The first 10 datas are: ', X[:10]) #EX2
        for i in range(5): #EX3
            print("Before __getitem__ the first five are: ", X[i]) #EX3
            print("__getitem__ returns: ", self.__getitem__(i)[0]) #EX3
        self.__len__() 
    
        

    def __len__(self):
        """
        Must return the length of the dataset, so the dataloader can know
        how to split it into batches

        Returns:
            (int): the length of the dataset
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Returns the _transformed_ item from the dataset
        
        Args:
            index (int):

        Returns:
            (tuple):
                * example (ndarray): vector representation of a training example
                * label (int): the class label
                * length (int): the length (tokens) of the sentence

        Examples:
            For an `index` where:
            ::
                self.data[index] = ['this', 'is', 'really', 'simple']
                self.target[index] = "neutral"

            the function will have to return something like:
            ::
                example = [  533  3908  1387   649   0     0     0     0]
                label = 1
                length = 4
        """
        
        
        example = np.zeros(60) #EX3
        
        if len(self.data[index]) < 60: #EX3
            length = len(self.data[index]) #EX3
            for i in range(length): #EX3
                
                if self.data[index][i] in self.word2idx: #EX3
                    example[i] = self.word2idx[self.data[index][i]] #EX3
                else: #EX3
                    example[i] = self.word2idx["<unk>"] #EX3
                
        else: #EX3
            length = 60 #EX3
            for i in range(60): #EX3
                if self.data[index][i] in self.word2idx: #EX3
                    example[i] = self.word2idx[self.data[index][i]] #EX3
                    
                else: #EX3
                    example[i] = self.word2idx["<unk>"] #EX3
                
        label = self.labels[index] #EX3
        
        return (example, label, length) #EX3

