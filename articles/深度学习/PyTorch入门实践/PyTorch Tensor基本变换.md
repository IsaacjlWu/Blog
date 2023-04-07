# PyTorch Tensor基本变换


刚接触机器学习的时候，初学者很想知道，经过了多层神经网络操作之后，我们的输入的`Tensor`的形状在每一步发生了什么改变？

同时，如果我们要对`Tensor`进行变换 **（升维、降维、变维等）**，有哪些常用的方法以及 API 呢？


**因此，本文梳理了两个重点如下 ⬇️**
- 1、Tensor的形状（shape）经过每一层神经网络层后发生的变化
- 2、改变Tensor的形状（shape）的方法以及API

---

## 神经网络操作中Tensor的变化

下面梳理了一下经过常见的神经网络层后，Tensor发生了什么改变～


# 全连接层
> 全连接操作是一种常用的神经网络操作，它将输入`Tensor`中的每个元素都连接到输出`Tensor`中的每个元素，使用**权重矩阵**对它们进行线性变换，并加上**偏置项**，最终生成一个新的`Tensor`。在这个过程中，`Tensor`的`shape`通常会发生变化。

上面的描述有点抽象，我用下面这张图来解释一下 ⬇️
<img src="../../../images/PyTorch/模型设计/全连接层示意图.png" width="600" />

其中每个神经元都会做一系列线性变换操作，再利用激活函数处理一下输出。可以看出，Tensor形状的变化，主要跟全连接层的神经元数量相关，而全连接层神经元数量则与全连接层的输出有关。

举个🌰

对于一个二维的输入Tensor，如果它的形状是 (batch_size, input_size)，那么全连接层的权重矩阵的形状通常是 (input_size, output_size)，其中 output_size 是输出的特征数量。全连接操作将输入Tensor的每个元素与权重矩阵相乘，并加上偏置项b，生成一个新的形状为 (batch_size, output_size) 的输出Tensor。



对于一个三维的输入张量，如果它的形状是 (batch_size, channels, input_size)，其中 channels 是输入的通道数，input_size 是输入的特征数量，那么全连接层的权重矩阵的形状通常是 (channels * input_size, output_size)。全连接操作将输入张量中的每个通道展开成一个向量，然后将这些向量连接在一起，生成一个形状为 (batch_size, channels * input_size) 的张量。接着，这个张量与权重矩阵相乘，并加上偏置项，生成一个新的形状为 (batch_size, output_size) 的输出张量。

总之，全连接操作可以改变张量的形状，并将输入张量中的每个元素与权重矩阵相乘，生成一个新的张量。在神经网络中，全连接操作通常用于将输入数据映射到更高维度的特征空间中，以便更好地提取和学习特征信息。

# 卷积层
卷积操作是一种常用的神经网络操作，它可以在输入的张量上滑动一个固定大小的窗口，并对窗口内的数据进行加权求和，生成一个新的张量。在这个过程中，张量的形状可能会发生变化，具体取决于所使用的卷积核的大小和步幅。

在二维卷积中，如果输入张量的形状是 (batch_size, channels, height, width)，卷积核的形状是 (out_channels, in_channels, kernel_height, kernel_width)，则输出张量的形状为 (batch_size, out_channels, output_height, output_width)，其中 output_height 和 output_width 是根据输入张量的大小、卷积核大小和步幅计算得出的。

在三维卷积中，如果输入张量的形状是 (batch_size, channels, depth, height, width)，卷积核的形状是 (out_channels, in_channels, kernel_depth, kernel_height, kernel_width)，则输出张量的形状为 (batch_size, out_channels, output_depth, output_height, output_width)，其中 output_depth、output_height 和 output_width 是根据输入张量的大小、卷积核大小和步幅计算得出的。

总之，卷积操作可以改变张量的形状和内容，生成一个新的张量，这个新的张量包含了输入张量中特定区域的加权和。在神经网络中，卷积操作可以帮助模型学习局部特征，并提取有用的信息。
# 池化层


池化操作是一种常用的特征降维方法，通常用于卷积神经网络中。池化操作将一个固定大小的窗口滑过输入的张量，然后对每个窗口内的元素进行聚合操作，例如取最大值或平均值等，得到一个新的张量。

池化操作会改变张量的形状，具体变化取决于池化的方式和参数设置。通常情况下，池化操作会减小张量的空间维度，例如减小宽度和高度，但保持深度（通道数）不变。举个例子，如果输入张量的形状为 [batch_size, height, width, channels]，则经过一个2x2的最大池化操作后，输出张量的形状会变成 [batch_size, height/2, width/2, channels]，其中height/2和width/2分别是输入张量的高度和宽度除以2。

需要注意的是，池化操作通常不会改变输入张量的批量大小（batch size），因此在输出张量的形状中，第一个维度的大小和输入张量保持一致。



另外，初学者还很想知道，如果我们要得到一个特定维度的Tensor（用于进行数学运算、计算损失的时候）应该用什么API对Tensor进行操作呢？

经常遇到的错误：
# mat1
# 3D Tensors
# 输入输出channel


# PyTorch改变tensor数据维度



在 pytorch 中，tensor 是基本的操作数据结构。在很多的时候，需要改变 tensor 的维度来适应咱们的计算，包括升维、降维、变维。在 pytorch 中有很多方法可以用来改变 tensor 的维度。
  
这里我把几种常用的方法进行了一下汇总：

view(shape)：返回一个新的 tensor，它具有给定的形状。如果元素总数不变，则可以用它来改变 tensor 的维度。例如：

import torch

t = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])
print(t.shape)  # torch.Size([2, 3])

t_view = t.view(3, 2)
print(t_view.shape)  # torch.Size([3, 2])
复制代码

unsqueeze(dim)：返回一个新的 tensor，它的指定位置插入了一个新的维度。例如：

import torch

t = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])
print(t.shape)  # torch.Size([2, 3])

t_unsqueeze = t.unsqueeze(0)
print(t_unsqueeze.shape)  # torch.Size([1, 2, 3])

t_unsqueeze = t.unsqueeze(1)
print(t_unsqueeze.shape)  # torch.Size([2, 1, 3])

t_unsqueeze = t.unsqueeze(2)
print(t_unsqueeze.shape)  # torch.Size([2, 3, 1])
复制代码

squeeze(dim)：返回一个新的 tensor，它的指定位置的维度的大小为 1 的维度被删除。例如：

import torch

t = torch.tensor([
    [[1], [2], [3]],
    [[4], [5], [6]]
])
print(t.shape)  # torch.Size([2, 3, 1])

t_squeeze = t.squeeze(2)
print(t_squeeze.shape)  # torch.Size([2, 3])

t_squeeze = t.squeeze()
print(t_squeeze.shape)  # torch.Size([2, 3])
复制代码

transpose(dim0, dim1)：返回一个新的 tensor，它的排列被交换。例如：

import torch

t = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])
print(t.shape)  # torch.Size([2, 3])

t_transpose = t.transpose(0, 1)
print(t_transpose.shape)  # torch.Size([3, 2])

t_transpose = t.transpose(1, 0)
print(t_transpose.shape)  # torch.Size([3, 2])
复制代码
另外还有一些其他的方法可以改变tensor的维度，例如 permute() 和 contiguous()。
好了，以上分享了 pytorch中改变tensor维度的方法，希望我的分享能对你的学习有一点帮助。


卷积神经网络（Convolutional Neural Network，CNN）是一种常用于图像处理和计算机视觉任务的神经网络。在CNN中，每个卷积层和全连接层的神经元数目的计算方法略有不同，下面分别介绍。

卷积层的神经元数目计算
在卷积层中，每个神经元只和上一层中某个局部区域的神经元相连。因此，卷积层的神经元数目取决于滤波器（Filter）的数量、滤波器的大小和输入数据的通道数。假设输入数据的形状为 [batch_size, height, width, channels]，卷积层使用的滤波器大小为 filter_size，滤波器的数量为 num_filters，则卷积层的神经元数目可以计算为：

神经元数目 = (height - filter_size + 2 * padding) / stride + 1) * (width - filter_size + 2 * padding) / stride + 1) * num_filters

其中，padding和stride分别表示填充和步幅参数。

全连接层的神经元数目计算
在全连接层中，每个神经元都和上一层中所有神经元相连。因此，全连接层的神经元数目只和上一层中的神经元数目有关。假设上一层有 n 个神经元，全连接层需要输出 m 个神经元，则全连接层的神经元数目可以计算为：

神经元数目 = m * n

综上所述，卷积神经网络的神经元数目取决于网络结构的设计、输入数据的形状以及每个层的参数设置。


