from typing import Dict

import torch
from torch.nn import Embedding, RNN, LSTM


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
        # self.model = RNN(input_size=embeddings.size(1), hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
        self.model = LSTM(input_size=embeddings.size(1), hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
        self.classify = torch.nn.Sequential(
            # torch.nn.Linear(1024,512),
            # torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden_size*2, hidden_size*2),
            torch.nn.ReLU(),
            # torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_size*2,num_class)
            )
    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        x = self.embed(batch)
        y, h_n = self.model(x)
        a, b, c = y.shape
        z = y.reshape((a,b, 2, -1))
        # print(z.shape)
        z = torch.cat((z[:, 0, 1,:],z[:, -1, 0,:]),1)
        z = z.view((a,-1))
        # print(z.shape)
        # z = z[:,512:-512]
        prediction = self.classify(z)
        return prediction
        raise NotImplementedError
    # def forward(self, batch) -> Dict[str, torch.Tensor]:
    #     # TODO: implement model forward
    #     x = self.embed(batch)
    #     y, h_n = self.model(x)
    #     print(h_n.shape)
    #     h_n = torch.permute(h_n, (1,0,2))
    #     a, b, c = h_n.shape
    #     h_n = h_n.reshape(a, -1)
    #     print(h_n.shape)
    #     prediction = self.classify(h_n)
    #     return prediction        
    #     # z = y.reshape((a,b, 2, -1))
        # # print(z.shape)
        # z = torch.cat((z[:, 0, 1,:],z[:, -1, 0,:]),1)
        # z = z.view((a,-1))
        # # print(z.shape)
        # # z = z[:,512:-512]
        # prediction = self.classify(z)
        # return prediction
        raise NotImplementedError


class SeqTagger(SeqClassifier):
    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        raise NotImplementedError
