# AT-DKT

## data preprocess
Please download the dataset and put them into the "data" folder, each dataset has one single folder in "data".
The original dataset of algebra2005 is provided already.
```
    cd examples/
    python data_preprocess.py --dataset_name=algebra2005
```

## train model

1. AT-DKT
```
    cd examples/
    # train algebra2005 & bridge2algebra2006
    python wandb_atdkt_train.py --use_wandb=0 --emb_type=qiddelxembhistranscembpredcurc --dataset_name=algebra2005
    # train nips_task34
    python wandb_atdkt_train.py --use_wandb=0 --emb_type=qiddelxembqqembhistranscembpredcurc --dataset_name=nips_task34
```
2. AT-DKT w/o IK
```
    cd examples/
    # train algebra2005 & bridge2algebra2006
    python wandb_atdkt_train.py --use_wandb=0 --emb_type=qiddelxembtranscembpredcurc --dataset_name=algebra2005
    # train nips_task34
    python wandb_atdkt_train.py --use_wandb=0 --emb_type=qiddelxembqqembtranscembpredcurc --dataset_name=nips_task34
```
3. AT-DKT w/o QT
```
    cd examples/
    # train algebra2005 & bridge2algebra2006
    python wandb_atdkt_train.py --use_wandb=0 --emb_type=qidcembpredhis --dataset_name=algebra2005
    # train nips_task34
    python wandb_atdkt_train.py --use_wandb=0 --emb_type=qidcembqembpredhis --dataset_name=nips_task34
```

## predict model
```
    cd examples/
    python wandb_predict.py --use_wandb=0 --save_dir=path/to/saved/model
```
