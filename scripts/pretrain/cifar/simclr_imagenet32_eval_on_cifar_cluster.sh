cd /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn
#git pull origin
DATA_PATH=/mnt/mmtech01/dataset/lzy/ILSVRC2012
#CIFAR_PATH=/mnt/mmtech01/dataset/wenzhuoliu/torch_ds
CIFAR_PATH=/mnt/mmtech01/usr/liuwenzhuo/torch_ds
DATASET=imagenet32
#    --val_data_path /share/wenzhuoliu/torch_ds/imagenet/val  \
# 0.075* sqrt(batch_size)
#    --weight_decay 1e-5 \ resnet18
#    --weight_decay 1e-6 \ resnet50
#    --eval_on_cifar \
/root/miniconda3/envs/solo-learn/bin/python main_pretrain.py \
    --dataset ${DATASET} \
    --backbone resnet50 \
    --train_data_path ${DATA_PATH}/train  \
    --val_data_path ${DATA_PATH}/val  \
    --cifar_path ${CIFAR_PATH} \
    --max_epochs 1000 \
    --devices 0,1,2,3,4,5,6,7 \
    --data_format dali \
    --accelerator gpu \
    --strategy ddp \
    --sync_batchnorm \
    --precision 16 \
    --optimizer lars \
    --grad_clip_lars \
    --eta_lars 0.02 \
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 2.4 \
    --classifier_lr 0.1 \
    --weight_decay 1e-6 \
    --batch_size 1024 \
    --num_workers 4 \
    --crop_size 32 \
    --brightness 0.8 \
    --contrast 0.8 \
    --saturation 0.8 \
    --hue 0.2 \
    --gaussian_prob 0.0 0.0 \
    --crop_size 32 \
    --num_crops_per_aug 1 1 \
    --name simclr-${DATASET} \
    --project solo-learn \
    --entity pigpeppa \
    --wandb \
    --save_checkpoint \
    --auto_resume \
    --method simclr \
    --temperature 0.2 \
    --proj_hidden_dim 2048 \
    --proj_output_dim 256

