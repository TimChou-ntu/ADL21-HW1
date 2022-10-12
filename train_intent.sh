export CUDA_VISIBLE_DEVICES=1
python train_elmo.py --data_dir=./data/intent/ --cache_dir=./cache/intent/ --ckpt_dir=./ckpt/intent/ --mode=intent --hidden_size=2048
python train_intent.py
bash intent_cls.sh ./data/intent/test.json pred_intent2.csv