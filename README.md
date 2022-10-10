# Sample Code for Homework 1 ADL NTU

## Environment
```shell
# If you have conda, we recommend you to build a conda environment called "adl-hw1"
make
conda activate adl-hw1
pip install -r requirements.txt
# Otherwise
pip install -r requirements.in
```

## Preprocessing
```shell
# To preprocess intent detectiona and slot tagging datasets
bash preprocess.sh
```

## Intent classification
```shell
python train_elmo.py --data_dir=./data/intent/ --cache_dir=./cache/intent/ --ckpt_dir=./ckpt/intent/ --mode=intent
python train_intent.py
bash inten_cls.sh
```

## Slot tagging
'''shell
python train_elmo.py --data_dir=./data/slot/ --cache_dir=./cache/slot/ --ckpt_dir=./ckpt/slot/ --mode=slot
python train_slot.py
bash slot_tag.sh
'''
