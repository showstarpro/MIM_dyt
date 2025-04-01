# 更新
加了dyt模块替换掉student里面（vit.py里面的注意力，教师的没改）的norm。设置了两个列表代表teacher和student做loss的层。loss有qk，vv，cls，dyt,dyt_f。（最后做加权即可）\
数据集方面由于imagenet太大所以是自己弄了个fake的，跑的时候直接注释掉和改路径用原本的就可以。\
环境方面直接用的最新版pytorch就可以，开始有点小的bug，但不多（网上一查就有）。\
pretrain添加了参数 --norm （输入dyt或norm）代表使用带dyt或者不带dyt的vit base，脚本处可以通过更改teacher_model来使用vit large或者vit base。\
finetune添加了参数 --norm，同上
