source /root/anaconda3/etc/profile.d/conda.sh

conda activate dytlpk

export WANDB_API_KEY=7352f9a349b74e01062672d0bc0bd3a8094677e2

torchrun --nproc_per_node 8 main_finetune.py
    --batch_size 128
    --epochs 300
    --data_path /lpai/dataset/imagenet-1k/0-1-0
    --lr 5e-4
    --weight_decay 0.05
    --num_workers 16
    --finetune MIM_dyt-master/test.pth
    --norm dyt