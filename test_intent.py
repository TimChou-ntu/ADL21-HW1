import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
import csv
from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len,train=False)
    # TODO: crecate DataLoader for test dataset
    test_dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=512)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqClassifier(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes,
    )
    model.eval()

    ckpt = torch.load(args.ckpt_path)
    # load weights into model
    model.load_state_dict(ckpt)
    model.to(args.device)

    with open(args.pred_file,'w',newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id","intent"])
        # TODO: predict dataset
        for idx, batch in enumerate(test_dataloader):
            batch['text'] = vocab.encode_batch([i.split() for i in batch['text']])
            batch['text'] = torch.Tensor(batch['text']).int().to(args.device)
            # batch['id'] = batch['id'].to(args.device)
            prediction = model(batch['text'])
            prediction = torch.argmax(prediction, dim=1)
            prediction = [dataset.idx2label(i) for i in prediction.tolist()]

        # TODO: write prediction to file (args.pred_file)
            for i in range(len(prediction)):
                writer.writerow([batch['id'][i],prediction[i].item()])

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")

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
