## Fine-tuning Pre-trained MAE for Classification


### Fine-tuning Pre-trained ChannelMAE

```
python submitit_finetune.py \
    --job_dir ${JOB_DIR} \
    --nodes 4 \
    --batch_size 32 \
    --model vit_ch_base_patch16 \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs 100 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path ${IMAGENET_DIR}
```


### Fine-tuning Pre-trained MAE

```
python submitit_finetune.py \
    --job_dir ${JOB_DIR} \
    --nodes 4 \
    --batch_size 32 \
    --model vit_base_patch16 \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs 100 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path ${IMAGENET_DIR}
```

### Linear Probing Pre-trained ChannelMAE

```
python submitit_linprobe.py \
    --job_dir ${JOB_DIR} \
    --nodes 4 \
    --batch_size 512 \
    --model vit_ch_base_patch16 --cls_token \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs 90 \
    --blr 0.1 \
    --weight_decay 0.0 \
    --dist_eval --data_path ${IMAGENET_DIR}
```


### Linear Probing Pre-trained MAE

```
python submitit_linprobe.py \
    --job_dir ${JOB_DIR} \
    --nodes 4 \
    --batch_size 512 \
    --model vit_base_patch16 --cls_token \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs 90 \
    --blr 0.1 \
    --weight_decay 0.0 \
    --dist_eval --data_path ${IMAGENET_DIR}
```
