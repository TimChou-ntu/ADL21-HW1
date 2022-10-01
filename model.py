from typing import Dict

import torch
from torch.nn import Embedding, RNN


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
        self.model = RNN(input_size=embeddings.size(1), hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
        self.classify = torch.nn.Sequential(
            # torch.nn.Linear(1024,512),
            # torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512,150)
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
        z = torch.cat((z[:, 0, 0,:],z[:, -1, 1,:]),1)
        z = z.view((a,-1))
        # print(z.shape)
        # z = z[:,512:-512]
        prediction = self.classify(z)
        return prediction
        raise NotImplementedError


class SeqTagger(SeqClassifier):
    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        raise NotImplementedError
