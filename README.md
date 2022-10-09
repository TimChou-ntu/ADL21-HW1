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
python train_elmo.py --data_dir=./data/intent/ --cache_dir=./cache/intent/ --ckpt_dir=./ckpt/intent/
python train_intent.py
python test_intent.py --test_file=./data/intent/test.json --ckpt_path=./previous_model_weight/gru....pt
```
