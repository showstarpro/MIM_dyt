torchrun --nproc_per_node 8 main_pretrain.py \
--batch_size 128 \
--epochs 100 \
--model tinymim_vit_base_patch16 \
--norm norm_dyt_dyt_dyt \
--data_path /lpai/dataset/imagenet-1k/0-1-0 \
--teacher_model mae_vit_base \
--teacher_path /lpai/inputs/models/dyt-dk-maevitbase/mae_pretrain_vit_base.pth \
--lr 5e-4 \
--weight_decay 0.05 \
--num_workers 16 \
--intermediate 4 8 12 \
--layer 4 8 12 \
--loss_weight 0 0 1 1 0 \
--output_dir /lpai/output/models/pre \
--log_dir /lpai/output/models/pre

torchrun --nproc_per_node 8 main_finetune.py \
--batch_size 128 \
--epochs 100 \
--data_path /lpai/dataset/imagenet-1k/0-1-0 \
--lr 5e-4 \
--weight_decay 0.05 \
--num_workers 16 \
--output_dir /lpai/output/models/fn \
--log_dir /lpai/output/models/fn \
--finetune /lpai/output/models/pre/checkpoint-99.pth \
--norm norm_dyt_dyt_dyt

torchrun --nproc_per_node 8 linear_prob.py \
--batch_size 128 \
--epochs 100 \
--data_path /lpai/dataset/imagenet-1k/0-1-0 \
--lr 5e-4 \
--weight_decay 0.05 \
--num_workers 16 \
--output_dir /lpai/output/models/ln \
--log_dir /lpai/output/models/ln \
--finetune /lpai/output/models/pre/checkpoint-99.pth \
--norm norm_dyt_dyt_dyt