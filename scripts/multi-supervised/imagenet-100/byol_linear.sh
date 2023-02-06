python3 main_linear.py \
    --dataset imagenet100 \
    --backbone resnet18 \
    --train_data_path /datasets/imagenet-100/train \
    --val_data_path /datasets/imagenet-100/val \
    --max_epochs 100 \
    --devices 0 \
    --accelerator gpu \
    --precision 16 \
    --optimizer sgd \
    --scheduler step \
    --lr 0.3 \
    --lr_decay_steps 60 80 \
    --weight_decay 0 \
    --batch_size 256 \
    --num_workers 4 \
    --data_format dali \
    --name byol-imagenet100-linear-eval \
    --pretrained_feature_extractor PATH \
    --project solo-learn \
    --entity unitn-mhug \
    --wandb \
    --save_checkpoint \
    --auto_resume