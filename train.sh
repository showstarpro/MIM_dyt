source /root/anaconda3/etc/profile.d/conda.sh

conda activate dytlpk

export WANDB_API_KEY=7352f9a349b74e01062672d0bc0bd3a8094677e2

torchrun --nproc_per_node 8 main_pretrain.py \
    --batch_size 128 \
    --epochs 100 \
    --data_path /lpai/dataset/imagenet-1k/0-1-0 \
    --output_dir /lpai/output/models \
    --teacher_path /lpai/MIM_dyt-master/mae_pretrain_vit_large.pth \
    --lr 5e-4 \
    --weight_decay 0.05 \
    --num_workers 16 \
    --intermediate 18 \
    --layer 12 \
    --loss_weight 0 0 1 1 1\
    --norm dyt\
    --teacher_model mae_vit_large