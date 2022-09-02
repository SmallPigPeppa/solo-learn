cd /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn
DATA_PATH=/mnt/mmtech01/dataset/lzy/ILSVRC2012
CIFAR_PATH=/mnt/mmtech01/usr/liuwenzhuo/torch_ds
DATASET=imagenet32

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
    --lr 0. \
    --classifier_lr 0.1 \
    --weight_decay 1e-6 \
    --batch_size 256 \
    --num_workers 4 \
    --crop_size 32 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.4 \
    --hue 0.1 \
    --gaussian_prob 0.0 0.0 \
    --crop_size 32 \
    --num_crops_per_aug 1 1 \
    --zero_init_residual \
    --name simsiam-${DATASET} \
    --project solo-learn \
    --entity pigpeppa \
    --wandb \
    --save_checkpoint \
    --auto_resume \
    --method simsiam \
    --proj_hidden_dim 2048 \
    --pred_hidden_dim 512 \
    --proj_output_dim 2048 \
    --resume_from_checkpoint /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/simsiam/22bn8hmt/simsiam-imagenet32-22bn8hmt-ep=999.ckpt
