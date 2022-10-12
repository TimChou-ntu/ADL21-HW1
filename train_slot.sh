# python train_elmo.py --data_dir=./data/slot/ --cache_dir=./cache/slot/ --ckpt_dir=./ckpt/slot/ --mode=slot --hidden_size=2048
python train_slot.py
bash slot_tag.sh ./data/slot/test.json pred_slot2.csv