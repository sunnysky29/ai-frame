# -*- coding: utf-8 -*-
"""
==============================================================================
Time : 2022/8/21 23:34
File : tensor_basics.py

YouTube 《Complete Pytorch Tensor Tutorial (Initializing Tensors, Math, Indexing, Reshaping)》：
        https://www.youtube.com/watch?v=x9JiIFvlUwk&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=2&ab_channel=AladdinPersson

==============================================================================
"""
import torch
# =========================================================================
#                                 tensor初始化
# =========================================================================
device = "cuda" if torch.cuda.is_available() else 'cpu'
my_tensor = torch.tensor([[12,5,7], [2,3,4]],
                         dtype=torch.float32,
                         device=device,
                         requires_grad=True)
print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.requires_grad)

# 其它初始化方法
x1 = torch.empty(size=(3,3))  # Returns a tensor filled with uninitialized data.
x = torch.zeros((3,3))
x = torch.rand((3,3))
x = torch.ones((3,3))
x = torch.eye(5,5)
x = torch.arange(start=0, end=5, step=1)
x= torch.linspace(start=0.1, end=1, steps=10)
x =torch.empty(size=(1, 5)).normal_(mean=0, std=1)
x = torch.empty(size=(1,5)).uniform_(0,1)
x= torch.diag(torch.ones(3))
print(x)

# types 转换
tensor = torch.arange(4)
print(tensor)
print(tensor.bool())
print(tensor.short())
print(tensor.long())
print(tensor.half())
print(tensor.double())
print(tensor)  # 原始未改变

# numpy 转换
import numpy as np
np_array = np.zeros((5,5))
tensor = torch.from_numpy(np_array)
np_back = tensor.numpy()

# 数学运算
x = torch.tensor([1,2,3])
y = torch.tensor([2,8,7])
print(f'x:{x}\n y:{y}')

z1 = torch.empty(3)
torch.add(x,y,out=z1)
z2= torch.add(x,y)
z3 = x+y
z4 = x -y
z5 = torch.true_divide(x,y)
print(z1, z2, z3, z4, z5)

# inplace operations
t = torch.zeros(3)
t.add_(x)
print(t)
t += x
print(t)

z = x.pow(2)
z = x **2
z = x > 0
z = x < 0
print(z)

# 矩阵运算
x1 = torch.rand((2,5))
x2 = torch.rand((5,3))
x3 = torch.mm(x1, x2)  # 2x3
print(x3)
x3= x1.mm(x2)
print(x3)

matrix_exp = torch.tensor([[1,2],
                           [3,4]])
print(matrix_exp)
print(matrix_exp.matrix_power(2))  # 矩阵幂
print(matrix_exp)  # 保持不变

# 元素相乘
z = x * y
print(z)  # tensor([ 9, 16, 21])
z = torch.dot(x,y)
print(z)

# batch 矩阵相乘
batch = 2
n = 1
m =2
p =3
tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))
print(tensor1, '\n', tensor2)
out_bmm = torch.bmm(tensor1, tensor2) # (batch, n, p)
print(out_bmm)

#  example of broadcasting
x1= torch.rand((5,5))
x2 = torch.rand((1,5))
print(f'x1:{x1} \n x2:{x2}')
z = x1-x2
print(z)
z = x1 ** x2
print(z)

x= torch.tensor([[2, -1, 3],
                 [3,4, 1]])
print(f'x:{x}')
sum_x0 = torch.sum(x, dim=0)
sum_x1 = torch.sum(x, dim=1)
print(sum_x0, sum_x1)
print(torch.abs(x))
v,i = torch.max(x, dim=1)
print(x.max(dim=1))  # 效果同上
print(v, i, sep='--&&')
z1 = torch.argmax(x, dim=1)
z2 = torch.argmin(x, dim=0)
print(z1, z2)
mean_x = torch.mean(x.float(), dim=0)
print(mean_x)
print(f'x:{x}')
print(f'y:{y}')
print(torch.eq(x,y))
a = torch.sort(y, dim=0, descending=False)
print(a)
print(f'y:{y}')

z = torch.clamp(x, min=2)
print(z)

x= torch.tensor([1,0,1,1], dtype=torch.bool)
z = torch.any(x)
z = torch.all(x)  # False
print(z)

# =========================================================================
#                                 tensor indexing
# =========================================================================
batch_size = 10
features =25
x= torch.rand((batch_size, features))
print(x[0])
print(x[0].shape)
print(x[:, 0].shape)
print(x[2, 0:3])  # [0,1,2]
x[0,0] = 10
# fancy indexing
x = torch.arange(10)
print(x)
indices = [2,5,8]
print(x[indices])

x = torch.rand((3,5))
print(f'x: {x}')
rows = torch.tensor([1,0])
cols = torch.tensor([4,0])
print(f'x[rows, cols]: {x[rows, cols]}')
print(x[rows, cols].shape)

x= torch.arange(10)
print(x)    # 0,1,2,..., 9
print(x[(x <2) | (x>8)])  # 逻辑运算
print(x[x.remainder(2)==0])
print(torch.where(x>5, x, x*2))  # 满足条件返回x, 否则返回 2*x
print(torch.tensor([0,0,1,2,3,2,4,1]).unique())
print(x.ndimension())   # 返回维度， 1
print(x.numel())


# =========================================================================
#                                 tensor reshape
# =========================================================================
x = torch.arange(9)

x_3x3 = x.view(3,3)
x_3x3 = x.reshape(3,3)
print(x_3x3)

y = x_3x3.t()  # 转置
print(y)
# print(y.view(9))  # 会报错
print(y.reshape(9))
print(y.contiguous().view(9))  # 会报错

x1 = torch.rand((2,5))
x2 = torch.rand((2,5))
print(torch.cat((x1,x2), dim=0))  # 4*5
print(torch.cat((x1,x2), dim=1))  # 2*10

z = x1.view(-1)
print(z)
batch = 64
x = torch.rand((batch, 2,5))
z =x.view(batch, -1)
print(z.shape)

z = x.permute(0,2,1)
print(z.shape)  # 维度互换

x = torch.arange(10)
print(x.unsqueeze(0).shape) # (1,10)
print(x.unsqueeze(1).shape)  # (10,1)

x= torch.arange(10).unsqueeze(0).unsqueeze(1)
print(x.shape)

z = x.squeeze(1)
print(z.shape)



