import torch
from torch import nn

input1 = torch.randn(4, 197, 768)
input2 = torch.randn(4, 197, 768)
cos = nn.CosineEmbeddingLoss(reduction='mean')

loss_flag = torch.ones([197]) # 需要初始化  一个N维的1或-1
output = 0.0
print(input1[1,:,:])


for i in range(4):
    output += cos(input1[i,:,:], input2[i,:,:], loss_flag)
    
output /= 4

print(output)	


