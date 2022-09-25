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

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        try:
            print(batch.size())
            print("tensor")
        except:
            print(len(batch))
            print("list")

        x = self.embed(batch)
        y = self.model(x)
        return y
        raise NotImplementedError


class SeqTagger(SeqClassifier):
    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        raise NotImplementedError
