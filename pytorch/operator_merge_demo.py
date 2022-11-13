# -*- coding: utf-8 -*-
"""
==============================================================================
Time : 2022/11/13 21:59
File : operator_merge_demo.py

算子融合
融合图：pic/operator_merge.jpeg

哔站视频：《16、PyTorch中进行卷积残差模块算子融合》
https://www.bilibili.com/video/BV1sU4y1u7TM/?spm_id_from=333.788&vd_source=abeb4ad4122e4eff23d97059cf088ab4
==============================================================================
"""

import torch
import torch.nn.functional as F
import torch.nn as nn

in_channels=2
out_channels = 2
kernel_size = 3
w = 9
h = 9
x = torch.ones(1, in_channels,w,h)  # 输入图片

# res_block = 3*3 conv + 1*1 conv + input

# ===========================
# 方法1：原生写法
# ===========================

conv_2d = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same')
conv_2d_pointwise = nn.Conv2d(in_channels, out_channels, 1)  # 1*1
result1 = conv_2d(x) + conv_2d_pointwise(x) + x
print(f'result1: {result1} \n {result1.shape}')

# ===========================
# 方法2：算子融合
# 把point-wise 卷积和x 本身都写成 3*3 的卷积
# 最终把三个卷积写成一个卷积
# ===========================
# 卷积维度扩充：2*2*1*1-> 2*2*3*3
pointwise_to_conv_weight = F.pad(conv_2d_pointwise.weight,
                                 [1,1,1,1,0,0,0,0])
# 构建实例，修改weight
conv2d_for_pointwise = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same')
conv2d_for_pointwise.weight = nn.Parameter(pointwise_to_conv_weight)
conv2d_for_pointwise.bias = conv_2d_pointwise.bias

# x改写成卷积形式
zeros = torch.unsqueeze(torch.zeros(kernel_size, kernel_size), 0) # 1*3*3
stars = torch.unsqueeze(F.pad(torch.ones(1,1), [1,1,1,1]), 0)  # 1*3*3， 中间为1，其余为0
stars_zeros = torch.unsqueeze(torch.cat([stars, zeros], 0), 0)   # 1*2*3*3
zeros_stars = torch.unsqueeze(torch.cat([zeros, stars], 0), 0)
identity_to_conv_weight = torch.cat([stars_zeros, zeros_stars], 0)  # 2*2*3*3
identity_to_conv_bias = torch.zeros([out_channels])
# 构建实例，修改weight
conv2d_for_identity = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same')
conv2d_for_identity.weight = nn.Parameter(identity_to_conv_weight)
conv2d_for_identity.bias = nn.Parameter(identity_to_conv_bias)

result2 = conv_2d(x)+ conv2d_for_pointwise(x) + conv2d_for_identity(x)
print(f'result2: {result2} \n {result2.shape}')
#比较结果是否一致
print(torch.all(torch.isclose(result1, result2)))

# 融合
conv_2d_for_fusion = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same')
conv_2d_for_fusion.weight = nn.Parameter(conv_2d.weight.data+conv2d_for_pointwise.weight.data+conv2d_for_identity.weight.data)
conv_2d_for_fusion.bias = nn.Parameter(conv_2d.bias.data+conv2d_for_pointwise.bias.data+conv2d_for_identity.bias.data)
result3 = conv_2d_for_fusion(x)
print(f'result3: {result3} \n {result3.shape}')
print(torch.all(torch.isclose(result3, result2)))