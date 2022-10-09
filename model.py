from typing import Dict

import torch
import torch.nn as nn
from torch.nn import Embedding, RNN, LSTM, GRU


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        # self.rnn = RNN(input_size=embeddings.size(1), hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional,batch_first=True)
        # self.rnn = LSTM(input_size=embeddings.size(1), hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional,batch_first=True)
        self.rnn = GRU(input_size=embeddings.size(1), hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional,batch_first=True)
        self.classify = torch.nn.Sequential(
            torch.nn.Linear(hidden_size*2,hidden_size*2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_size*2,hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_size,hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_size,num_class)
            )
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias,0)

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        x = self.embed(batch)
        y, h_n = self.rnn(x)
        a, b, c = y.shape
        z = y.reshape((a,b, 2, -1))
        z = torch.cat((z[:, 0, 1,:],z[:, -1, 0,:]),1)
        prediction = self.classify(z)
        return prediction
        raise NotImplementedError


class SeqTagger(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqTagger, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        self.rnn = GRU(input_size=embeddings.size(1), hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional,batch_first=True)
        self.classify = torch.nn.Sequential(
            torch.nn.Linear(hidden_size*2,hidden_size*2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_size*2,hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_size,hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_size,num_class)
            )
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias,0)


    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        x = self.embed(batch)
        y, h_n = self.rnn(x)
        a, b, c = y.shape
        per_token_prediction = self.classify(y)
        return per_token_prediction

        raise NotImplementedError
