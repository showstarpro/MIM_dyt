# TinyMIM with DyT & Distillation

## 😎 Introduction
框架为TinyMIM的框架，其中替换了：
1. Tiny源码的利用GPU训练部分，支持在本地用CPU跑
2. 原始数据集ImageNet，从tiny-imagenet-200中提取了1class * 10 images，下载代码见download.py
3. student中的norm，改为DyT，使用了 Transformer Without Normalization(https://arxiv.org/abs/2503.10622) 中的DyT代码。支持使用命令行参数dynamic_tanh值为true/false，选择加/不加dyt
4. 原来的qkloss, vvloss，替换为cls_token loss，可以选择对teacher中的12,15,18,21,24层或整体蒸馏，也可以选择对每个tranformer块的第一个DyT、第二个DyT后的输出、或整体进行蒸馏。可以利用命令行参数distill_layer (6,8,9,11,12,all), distill_dyt (before,after,all)控制
5. 训练模型时需要加载的mae_vit_large，作为teacher模型的参数，由于文件过大，没有上传