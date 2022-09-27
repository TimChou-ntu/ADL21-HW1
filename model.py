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
        print(embeddings.size())
        self.model = RNN(input_size=embeddings.size(1), hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
        self.classify = torch.nn.Sequential(
            torch.nn.Linear(131072, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1024,512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(131072, 1024),
            torch.nn.ReLU(inplace=True),
            )
    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        print(batch.shape)
        x = self.embed(batch)
        print(x.shape)
        y, h_n = self.model(x)
        print(y.shape)
        print(h_n.shape)
        return y
        raise NotImplementedError


class SeqTagger(SeqClassifier):
    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        raise NotImplementedError
