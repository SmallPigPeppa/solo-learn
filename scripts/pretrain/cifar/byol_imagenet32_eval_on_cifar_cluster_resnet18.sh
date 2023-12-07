#cd /mnt/mmtech01/usr/liuwenzhuo/code/solo-learn
DATA_PATH=/mnt/mmtech01/dataset/lzy/ILSVRC2012
CIFAR_PATH=/mnt/mmtech01/usr/liuwenzhuo/torch_ds
DATASET=imagenet32
#    --eval_on_cifar \
#    --cifar_path ${CIFAR_PATH} \
/root/miniconda3/envs/solo-learn/bin/python main_pretrain.py \
    --dataset ${DATASET} \
    --backbone resnet18 \
    --train_data_path ${DATA_PATH}/train  \
    --val_data_path ${DATA_PATH}/val  \
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
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 2.0 \
    --accumulate_grad_batches 16 \
    --classifier_lr 0.1 \
    --weight_decay 1e-5 \
    --batch_size 512 \
    --num_workers 4 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.2 \
    --hue 0.1 \
    --gaussian_prob 0.0 0.0 \
    --solarization_prob 0.0 0.2 \
    --crop_size 32 \
    --num_crops_per_aug 1 1 \
    --name byol-resnet18-${DATASET} \
    --project solo-learn \
    --entity pigpeppa \
    --wandb \
    --save_checkpoint \
    --auto_resume \
    --method byol \
    --proj_output_dim 256 \
    --proj_hidden_dim 4096 \
    --pred_hidden_dim 4096 \
    --base_tau_momentum 0.99 \
    --final_tau_momentum 1.0 \
    --momentum_classifier
