cd /lpai ;

git clone -b LPK https://github.com/showstarpro/MIM_dyt.git MIM_dyt ;

cd ./MIM_dyt;

source /root/anaconda3/etc/profile.d/conda.sh ;

conda activate dytlpk;
export WANDB_API_KEY=7352f9a349b74e01062672d0bc0bd3a8094677e2;

torchrun --nproc_per_node 8 main_pretrain.py \
    --batch_size 128 \
    --epochs 100 \
    --model tinymim_vit_base_patch16 \
    --data_path /lpai/dataset/imagenet-1k/0-1-0 \
    --teacher_path /lpai/MIM_dyt-master/mae_pretrain_vit_large.pth \
    --lr 5e-4 \
    --weight_decay 0.05 \
    --num_workers 16 \
    --intermediate 18 \
    --layer 12 \
    --loss_weight 0 0 1 1 0\
    --norm norm\
    --teacher_model mae_vit_large\
    --output_dir /lpai/output/models/stu_norm/pre \
    --log_dir  /lpai/output/models/stu_norm/pre ;


torchrun --nproc_per_node 8 main_finetune.py \
    --batch_size 128 \
    --epochs 100 \
    --data_path /lpai/dataset/imagenet-1k/0-1-0 \
    --lr 5e-4 \
    --weight_decay 0.05 \
    --num_workers 16 \
    --output_dir /lpai/output/models/stu_norm/fn \
    --log_dir  /lpai/output/models/stu_norm/fn \
    --finetune /lpai/output/models/stu_norm/pre/checkpoint-99.pth \
    --norm norm ;

torchrun --nproc_per_node 8 linear_prob.py \
    --batch_size 128 \
    --epochs 100 \
    --data_path /lpai/dataset/imagenet-1k/0-1-0 \
    --lr 5e-4 \
    --weight_decay 0.05 \
    --num_workers 16 \
    --output_dir /lpai/output/models/stu_norm/ln \
    --log_dir  /lpai/output/models/stu_norm/ln \
    --finetune /lpai/output/models/stu_norm/pre/checkpoint-99.pth \
    --norm norm ;

sleep 14d;

