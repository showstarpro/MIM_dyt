# 更新
就加了dyt模块替换掉student里面（vit.py里面的注意力，教师的没改）的norm，然后做了原本teacher（large）18层和student（base）第12层的dyt_loss。
数据集方面由于imagenet太大所以是自己弄了个fake的，跑的时候直接注释掉和改路径用原本的就可以。
环境方面直接用的最新版pytorch就可以，开始有点小的bug，但不多（网上一查就有）。
