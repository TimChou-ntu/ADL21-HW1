import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from random import shuffle
from typing import Dict

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from dataset import SeqTaggingClsDataset
from model import Elmo_embedding, SeqTagger
from utils import Vocab

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

# ACC
def count_acc(prediction, label, seq_len, joint=True):
    pred = torch.argmax(prediction, dim=1)
    index = 0
    acc = 0
    basis = max(seq_len)
    for i in seq_len:
        acc += sum((pred[index:index+i] == label[index:index+i]).tolist())
        index += basis    
    return 100*acc/sum(seq_len)





def main(args):
    # TODO: implement main function
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)
    tags2idx_path = args.cache_dir / "tag2idx.json"
    tags2idx: Dict[str,int] = json.loads(tags2idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqTaggingClsDataset] = {
        split: SeqTaggingClsDataset(split_data, vocab, tags2idx, args.max_len)
        for split, split_data in data.items()
    }
    # class num
    num_classes = len(vocab.token2idx)
    # Dataloader
    train_dataloader = torch.utils.data.DataLoader(datasets[TRAIN],batch_size=args.batch_size,collate_fn=datasets[TRAIN].collate_fn,shuffle=True)
    eval_dataloader = torch.utils.data.DataLoader(datasets[DEV],batch_size=512,collate_fn=datasets[DEV].collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # model
    model = Elmo_embedding(embeddings=embeddings, hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout, bidirectional=args.bidirectional, num_class=num_classes)
    model.to(args.device)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,100,150], gamma=0.1)

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    best_acc = 0
    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        # TRAIN
        model.train()
        total_loss1 = 0
        total_loss2 = 0
        total_acc1 = []
        total_acc2 = []
        for idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            loss = None
            prediction = None            

            batch['tokens'] = batch['tokens'].to(args.device)
            prediction1, prediction2 = model(batch["tokens"])
            input_tokens = batch['tokens'].reshape(-1).long()
            prediction1 = prediction1.reshape(-1, num_classes)
            prediction2 = prediction2.reshape(-1, num_classes)
            p1 = torch.Tensor([]).float().to(args.device)
            p2 = torch.Tensor([]).float().to(args.device)
            t1 = torch.Tensor([]).long().to(args.device)
            t2 = torch.Tensor([]).long().to(args.device)
            index = 0
            basis = max(batch['seq_len'])
            for length in batch['seq_len']:
                p1 = torch.cat((p1,prediction1[index:index+length-1,:]),dim=0)
                p2 = torch.cat((p2,prediction2[index+1:index+length,:]),dim=0)
                t1 = torch.cat((t1,input_tokens[index+1:index+length]),dim=0)
                t2 = torch.cat((t2,input_tokens[index:index+length-1]),dim=0)
                index += basis
            loss1 = criterion(p1,t1)
            loss2 = criterion(p2,t2)
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()

            total_loss1 += loss1.item()
            total_loss2 += loss2.item()
            acc1 = count_acc(p1, t1,batch['seq_len'])
            acc2 = count_acc(p2, t2,batch['seq_len'])
            total_acc1.append(acc1)
            total_acc2.append(acc2)
        lr_scheduler.step()


        print("Training acc1: %2.3f" %(sum(total_acc1)/len(total_acc1)),"Training acc2: %2.3f" %(sum(total_acc2)/len(total_acc2)), "Training Loss: %1.3f"%(10000*total_loss1/len(datasets[TRAIN])), "Training Loss: %1.3f"%(10000*total_loss2/len(datasets[TRAIN])))
        # EVAL
        # TODO: Evaluation loop - calculate accuracy and save model weights
        with torch.no_grad():
            model.eval()
            total_acc1 = []
            total_acc2 = []
            total_loss1 = 0
            total_loss2 = 0
            for idx, batch in enumerate(eval_dataloader):
                prediction = None            
                batch['tokens'] = batch['tokens'].to(args.device)
                prediction1, prediction2 = model(batch["tokens"])
                input_tokens = batch['tokens'].reshape(-1).long()
                prediction1 = prediction1.reshape(-1, num_classes)
                prediction2 = prediction2.reshape(-1, num_classes)
                p1 = torch.Tensor([]).float().to(args.device)
                p2 = torch.Tensor([]).float().to(args.device)
                t1 = torch.Tensor([]).long().to(args.device)
                t2 = torch.Tensor([]).long().to(args.device)
                index = 0
                basis = max(batch['seq_len'])
                for length in batch['seq_len']:
                    p1 = torch.cat((p1,prediction1[index:index+length-1,:]),dim=0)
                    p2 = torch.cat((p2,prediction2[index+1:index+length,:]),dim=0)
                    t1 = torch.cat((t1,input_tokens[index+1:index+length]),dim=0)
                    t2 = torch.cat((t2,input_tokens[index:index+length-1]),dim=0)
                    index += basis
                loss1 = criterion(p1,t1)
                loss2 = criterion(p2,t2)
                loss = loss1 + loss2

                total_loss1 += loss1.item()
                total_loss2 += loss2.item()
                acc1 = count_acc(p1, t1,batch['seq_len'])
                acc2 = count_acc(p2, t2,batch['seq_len'])
                total_acc1.append(acc1)
                total_acc2.append(acc2)
            
            acc1 = sum(total_acc1)/len(total_acc1)
            acc2 = sum(total_acc2)/len(total_acc2)
            print("Evaluate acc1:%1.3f" %acc1,"Evaluate acc2:%1.3f" %acc2, "Evaluate Loss1:%2.3f"%(10000*total_loss1/len(datasets[DEV])), "Evaluate Loss2:%2.3f"%(10000*total_loss2/len(datasets[DEV])))
            if acc1+acc2 > best_acc:
                torch.save(model.state_dict(),"./elmo.pt")
                best_acc = acc1+acc2


    raise NotImplementedError


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
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
        default="./ckpt/slot/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
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