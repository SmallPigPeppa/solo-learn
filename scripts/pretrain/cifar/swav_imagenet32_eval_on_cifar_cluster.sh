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
    --cifar_path ${CIFAR_PATH} \
    --eval_on_cifar \
    --max_epochs 1000 \
    --devices 0,1,2,3,4,5,6,7 \
    --accelerator gpu \
    --data_format dali \
    --strategy ddp \
    --sync_batchnorm \
    --precision 16 \
    --optimizer lars \
    --grad_clip_lars \
    --eta_lars 0.02 \
    --scheduler warmup_cosine \
    --lr 0.6 \
    --min_lr 0.0006 \
    --classifier_lr 0.1 \
    --weight_decay 1e-6 \
    --batch_size 128 \
    --num_workers 4 \
    --crop_size 32 \
    --brightness 0.8 \
    --contrast 0.8 \
    --saturation 0.8 \
    --hue 0.2 \
    --gaussian_prob 0.0 0.0 \
    --crop_size 32 \
    --num_crops_per_aug 1 1 \
    --name swav-${DATASET} \
    --project solo-learn \
    --entity pigpeppa \
    --wandb \
    --save_checkpoint \
    --auto_resume \
    --method swav \
    --proj_hidden_dim 2048 \
    --queue_size 3840 \
    --proj_output_dim 128 \
    --num_prototypes 3000 \
    --epoch_queue_starts 50 \
    --freeze_prototypes_epochs 2
