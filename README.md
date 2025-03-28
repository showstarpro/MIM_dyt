# 更新
加了dyt模块替换掉student里面（vit.py里面的注意力，教师的没改）的norm。设置了两个列表代表teacher和student做loss的层。loss有qk，vv，cls，dyt,dyt_f。（最后做加权即可）
数据集方面由于imagenet太大所以是自己弄了个fake的，跑的时候直接注释掉和改路径用原本的就可以。
环境方面直接用的最新版pytorch就可以，开始有点小的bug，但不多（网上一查就有）。
加了finetune功能
pretrain：python MIM_dyt-master/MIM_dyt-master/main_pretrain.py --batch_size 128  --epochs 300 --data_path lpai/dataset/imagenet-1k/0-1-0  --lr 5e-4  --weight_decay 0.05 --num_workers 0 --intermediate [18] --layer [12] --loss_weight [0,0,1,1,1]
finetune：python MIM_dyt-master/MIM_dyt-master/main_finetune.py --batch_size 128  --epochs 300 --data_path lpai/dataset/imagenet-1k/0-1-0  --lr 5e-4  --weight_decay 0.05 --num_workers 0 --finetune output_dir/checkpoint-0.pth
