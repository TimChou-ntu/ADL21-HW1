import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from tqdm import trange

from dataset import SeqClsDataset
from utils import Vocab
from model import SeqClassifier

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def count_acc(prediction, label):
    pred = torch.argmax(prediction, dim=1)
    return 100*((pred == label).type(torch.cuda.FloatTensor).mean().item())



def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    # TODO: crecate DataLoader for train / dev datasets
    train_dataloader = torch.utils.data.DataLoader(dataset=datasets[TRAIN], batch_size=args.batch_size,shuffle=True,collate_fn=datasets[TRAIN].collate_fn)
    eval_dataloader = torch.utils.data.DataLoader(dataset=datasets[DEV], batch_size=512,collate_fn=datasets[DEV].collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqClassifier(embeddings=embeddings,hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout, bidirectional=args.bidirectional, num_class=datasets[TRAIN].num_classes)
    model.to(args.device)
    criterion = torch.nn.CrossEntropyLoss()
    # TODO: init optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,300], gamma=0.1)

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    best_acc = 0
    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        # TRAIN
        model.train()
        total_loss = 0
        total_acc = []
        for idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            loss = None
            prediction = None            

            batch['text'] = batch['text'].int().to(args.device)
            batch['intent'] = batch['intent'].to(args.device)
            prediction = model(batch["text"])

            loss = criterion(prediction,batch['intent'])
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            acc = count_acc(prediction, batch['intent'])
            total_acc.append(acc)
        # lr_scheduler.step()


        print("Training acc: %2.3f" %(sum(total_acc)/len(total_acc)), "Training Loss: %1.3f"%(100*total_loss/len(datasets[TRAIN])))
        # EVAL
        # TODO: Evaluation loop - calculate accuracy and save model weights
        with torch.no_grad():
            model.eval()
            total_acc = []
            total_loss = 0
            for idx, batch in enumerate(eval_dataloader):
                prediction = None            
                batch['text'] = batch['text'].to(args.device)
                batch['intent'] = batch['intent'].to(args.device)
                prediction = model(batch["text"])

                total_loss += criterion(prediction, batch["intent"]).item()
                acc = count_acc(prediction, batch['intent'])
                total_acc.append(acc)
            
            acc = sum(total_acc)/len(total_acc)
            print("Evaluate acc:%1.3f" %acc, "Evaluate Loss:%2.3f"%(100*total_loss/len(datasets[DEV])))
            if acc > best_acc:
                torch.save(model.state_dict(),"./best.pt")
                best_acc = acc



        pass

    # TODO: Inference on test set



def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-2)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=300)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
