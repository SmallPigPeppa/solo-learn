cd /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn
DATA_PATH=/mnt/mmtech01/dataset/lzy/ILSVRC2012
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
    --eval_on_cifar \
    --cifar_path ${CIFAR_PATH} \
    --max_epochs 1000 \
    --devices 0,1,2,3,4,5,6,7 \
    --accelerator gpu \
    --data_format dali \
    --strategy ddp \
    --sync_batchnorm \
    --precision 16 \
    --optimizer sgd \
    --scheduler warmup_cosine \
    --lr 0.3 \
    --classifier_lr 0.3 \
    --weight_decay 1e-6 \
    --batch_size 512 \
    --num_workers 4 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.4 \
    --hue 0.1 \
    --gaussian_prob 0.0 0.0 \
    --crop_size 32 \
    --num_crops_per_aug 1 1 \
    --name mocov2plus-${DATASET} \
    --project solo-learn \
    --entity pigpeppa \
    --wandb \
    --save_checkpoint \
    --auto_resume \
    --method mocov2plus \
    --proj_hidden_dim 2048 \
    --queue_size 32768 \
    --temperature 0.2 \
    --base_tau_momentum 0.99 \
    --final_tau_momentum 0.999 \
    --momentum_classifier
