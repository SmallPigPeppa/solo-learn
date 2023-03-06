python3 main_pretrain.py \
    --dataset cifar100 \
    --backbone resnet18 \
    --train_data_path /home/admin/torch_ds \
    --val_data_path /home/admin/torch_ds \
    --max_epochs 100 \
    --devices 0,1 \
    --accelerator gpu \
    --precision 16 \
    --num_workers 4 \
    --optimizer lars \
    --grad_clip_lars \
    --eta_lars 0.02 \
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 0.3 \
    --weight_decay 1e-4 \
    --batch_size 256 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.2 \
    --hue 0.1 \
    --gaussian_prob 0.0 0.0 \
    --solarization_prob 0.0 0.2 \
    --crop_size 32 \
    --num_crops_per_aug 1 1 \
    --name barlow-cifar10 \
    --project solo-learn-cifar10 \
    --entity pigpeppa \
    --wandb \
    --save_checkpoint \
    --auto_resume \
    --method barlow_twins \
    --proj_hidden_dim 2048 \
    --proj_output_dim 2048 \
    --scale_loss 0.1
