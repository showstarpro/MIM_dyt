import torch 

a = torch.tensor([[1,2],[3,4]])
b = torch.tensor([[5,6],[7,8]])
c = []
c.append(a)
c.append(b)
print(c)

print(torch.tensor(c))

print(int('11') == 11)