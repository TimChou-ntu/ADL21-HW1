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
from model import SeqTagger, Elmo_embedding
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

    num_classes_vocab = len(vocab.token2idx)

    model = SeqTagger(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes,
    )
    elmo = Elmo_embedding(embeddings=embeddings, hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout, bidirectional=args.bidirectional, num_class=num_classes_vocab)

    model.eval()
    elmo.eval()

    ckpt_m = torch.load(args.ckpt_dir)
    # load weights into model
    model.load_state_dict(ckpt_m)
    model.to(args.device)

    ckpt_e = torch.load(args.ckpt_dir / 'elmo.pt')
    elmo.load_state_dict(ckpt_e)
    elmo.to(args.device)

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
            p1, p2, elmo_embedding = elmo(batch['tokens'])
            prediction = model((batch["tokens"], elmo_embedding))
            prediction = prediction.reshape(-1, num_classes)

        # TODO: write prediction to file (args.pred_file)
            pred = torch.argmax(prediction, dim=1)
            index = 0
            basis = max(batch['seq_len'])
            for id, i in enumerate(batch['seq_len']):
                pred_tag = " ".join([dataset.idx2label(i) for i in (pred[index:index+i].tolist())])
                writer.writerow([batch['id'][id],pred_tag])
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
        default="./ckpt/slot/slot.pt",
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