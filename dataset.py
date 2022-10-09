from typing import List, Dict

from torch.utils.data import Dataset
import torch

from utils import Vocab


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
        train=True
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len
        self.train = train



    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        text = [i['text'].split() for i in samples]
        text = self.vocab.encode_batch(text)
        id = [i['id'] for i in samples]
        if self.train:
            intent = [self.label2idx(i['intent']) for i in samples]
            return {"text":torch.Tensor(text).long(),"id":id,"intent":torch.Tensor(intent).long()}
        else:
            return {"text":torch.Tensor(text).long(),"id":id}

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SeqTaggingClsDataset(SeqClsDataset):
    ignore_idx = -100

    def collate_fn(self, samples):
        # TODO: implement collate_fn
        tokens = [i["tokens"] for i in samples]
        seq_len = [len(i) for i in tokens]
        tokens = self.vocab.encode_batch(tokens)
        id = [i["id"] for i in samples]
        if self.train:
            tags = torch.nn.utils.rnn.pad_sequence([torch.Tensor([self.label2idx(x) for x in i["tags"]]) for i in samples],batch_first=True, padding_value=-1.0)
            return {"tokens":torch.Tensor(tokens).long(),"id":id,"seq_len":seq_len,"tags":torch.Tensor(tags).long()}
        else:
            return {"tokens":torch.Tensor(tokens).long(),"id":id,"seq_len":seq_len}



        raise NotImplementedError
