import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class CBOW(nn.Module):
    # predict a masked word (in the middle) given the context around it (window of size x)
    def __init__(self, embedding_size, vocabulary_size):
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.projection = nn.Linear(embedding_size, 64)
        self.output = nn.Linear(64, vocabulary_size)

    def forward(self, input):
        # We sum up the input vectors (context around the target word)
        embedded = sum(self.embedding(input)).view(1, -1)
        projection = F.relu(self.projection(embedded))
        return F.log_softmax(self.output(projection))



