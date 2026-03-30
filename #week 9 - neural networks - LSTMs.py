#notebook practice

# pip install d21 --- not working in positron

#let's use Dracula as our source text:
#https://www.gutenberg.org/cache/epub/345/pg345.txt

#for reading/cleaning data
import requests
import re
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

#for visualizing
import plotly.express as px

#for model building
import torch
import torch.nn as nn
import torch.nn.functional as F

#import helpers
import nnhelpers as nnh
from d2l import torch as d2l

#customized version of the d2l Time Machine class object,
#   but using a better source text
class Dracula(d2l.DataModule):
    def _download(self):
        dracula = "https://www.gutenberg.org/cache/epub/345/pg345.txt"
        fname = d2l.download(dracula, self.root)
        with open(fname, encoding='utf-8')as f:
            return f.read()
    
    def _preprocess(self, text):
        """Defined in :numref:'sec_text-sequence'"""
        return re.sub('[^A-Za-z]+', ' ', text).lower()
    
    def _tokenize(self, text):
        """Defined in :numref:'sec_text-sequence'"""
        return list(text)
    
    def build(self, raw_text, vocab=None):
        """Defined in :numref:`sec_text-sequence`"""
        tokens = self._tokenize(self._preprocess(raw_text))
        if vocab is None: vocab = d2l.Vocab(tokens)
        corpus = [vocab[token] for token in tokens]
        return corpus, vocab

    def __init__(self, batch_size, num_steps, num_train=10000, num_val=5000):
        """Defined in :numref:`sec_language-model`"""
        super(Dracula, self).__init__()
        self.save_hyperparameters()
        corpus, self.vocab = self.build(self._download())
        array = d2l.tensor([corpus[i:i+num_steps+1]
                            for i in range(len(corpus)-num_steps)])
        self.X, self.Y = array[:,:-1], array[:,1:]

    def get_dataloader(self, train):
        """Defined in :numref:`subsec_partitioning-seqs`"""
        idx = slice(0, self.num_train) if train else slice(
            self.num_train, self.num_train + self.num_val)
        return self.get_tensorloader([self.X, self.Y], train, idx)

class LSTM(d2l.Module):
    def __init__(self, num_inputs, num_hiddens, 
                sigma=0.01, lr=3, numeric=False):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.numeric = numeric
        init_weight = lambda *shape: nn.Parameter(
            torch.randn(*shape) * sigma)
        triple = lambda: (init_weight(num_inputs, num_hiddens),
                          init_weight(num_hiddens, num_hiddens),
                          nn.Parameter(torch.zeros(num_hiddens)))
        self.W_xi, self.W_hi, self.b_i = triple()  # Input gate
        self.W_xf, self.W_hf, self.b_f = triple()  # Forget gate
        self.W_xo, self.W_ho, self.b_o = triple()  # Output gate
        self.W_xc, self.W_hc, self.b_c = triple()  # Input node

    def forward(self, inputs, H_C=None):
        if H_C is None:
            # Initial state with shape: (batch_size, num_hiddens)
            H = torch.zeros((inputs.shape[1], self.num_hiddens),
                          device=inputs.device)
            C = torch.zeros((inputs.shape[1], self.num_hiddens),
                          device=inputs.device)
        else:
            H, C = H_C
        outputs = []
        for X in inputs:
            I = torch.sigmoid(torch.matmul(X, self.W_xi) +
                            torch.matmul(H, self.W_hi) + self.b_i)
            F = torch.sigmoid(torch.matmul(X, self.W_xf) +
                            torch.matmul(H, self.W_hf) + self.b_f)
            O = torch.sigmoid(torch.matmul(X, self.W_xo) +
                            torch.matmul(H, self.W_ho) + self.b_o)
            C_tilde = torch.tanh(torch.matmul(X, self.W_xc) +
                              torch.matmul(H, self.W_hc) + self.b_c)
            C = F * C + I * C_tilde
            H = O * torch.tanh(C)
            outputs.append(H)
        return outputs, (H, C)

#running the model
data = Dracula(batch_size=1024, num_steps=100)
lstm = LSTM(num_inputs=len(data.vocab), num_hiddens=128)
model = d2l.RNNLMScratch(lstm, vocab_size=len(data.vocab), lr=4)
trainer = d2l.Trainer(max_epochs=50, gradient_clip_val=1, num_gpus=1)
trainer.fit(model, data)

#making predictions
model.predict('as the boat arrived at the quay ', 
    50, data.vocab, d2l.try_gpu())
#output = as the boat arrived at the quay the sound the sound the sound the sound the sound'
#could be better but this is the prediction it comes across