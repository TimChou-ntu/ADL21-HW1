import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import csv

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag2idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag2idx_path.read_text())

    data = json.loads(args.data_dir.read_text())
    dataset = SeqTaggingClsDataset(data, vocab, tag2idx, args.max_len,train=False)
    # TODO: crecate DataLoader for test dataset
    test_dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=512,collate_fn=dataset.collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqTagger(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes,
    )
    model.eval()

    ckpt = torch.load(args.ckpt_dir)
    # load weights into model
    model.load_state_dict(ckpt)
    model.to(args.device)

    # class num
    num_classes = dataset.num_classes


    with open(args.pred_file,'w',newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id","tags"])
        # TODO: predict dataset
        for idx, batch in enumerate(test_dataloader):
            prediction = 0
            pred = 0
            batch['tokens'] = batch['tokens'].to(args.device)
            prediction = model(batch["tokens"])
            prediction = prediction.reshape(-1, num_classes)

        # TODO: write prediction to file (args.pred_file)
            pred = torch.argmax(prediction, dim=1)
            index = 0
            basis = max(batch['seq_len'])
            for idx, i in enumerate(batch['seq_len']):
                pred = " ".join([dataset.idx2label(i) for i in (pred[index:index+i].tolist())])
                writer.writerow([batch['id'][idx],pred])
                index += basis



def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/test.json",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./slot.pt",
    )
    parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)