import pandas as pd
import numpy as np
import re


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt

torch.autograd.set_detect_anomaly(True)

class PhraseEmbedding(nn.Module):
    def __init__(self):
	    super(NN, self).__init__()

# 	def __init__(self,input_size,output_size):
# 	    super(NN, self).__init__()
# 	    self.dense_layer = nn.Linear(input_size,output_size)
      
	def forward(self, input):
	    print(input.shape)
# 		x = self.dense_layer(x)
# 		out = nn.Tanh(x)
# 		return out

	def getEmbedding(self, input_embeddings):
		with torch.no_grad():
			return self.forward(input_embeddings)