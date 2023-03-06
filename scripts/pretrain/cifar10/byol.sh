python3 main_pretrain.py \
    --dataset cifar10 \
    --backbone resnet18 \
    --train_data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds \
    --val_data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds \
    --max_epochs 100 \
    --devices 0 \
    --accelerator gpu \
    --precision 16 \
    --optimizer lars \
    --grad_clip_lars \
    --eta_lars 0.02 \
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 1.0 \
    --classifier_lr 0.1 \
    --weight_decay 1e-5 \
    --batch_size 256 \
    --num_workers 4 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.2 \
    --hue 0.1 \
    --gaussian_prob 0.0 0.0 \
    --solarization_prob 0.0 0.2 \
    --crop_size 32 \
    --num_crops_per_aug 1 1 \
    --name byol-cifar10 \
    --project solo-learn-cifar10 \
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
