# PyTorch指标可视化

# visdom hello world示例
```python
import visdom
import numpy as np
```

Visdom是一种用于实时数据可视化的Python库，特别适用于PyTorch深度学习框架。它可以用于实时监视和可视化模型的训练和评估指标，例如损失函数和准确度。

以下是在PyTorch中使用Visdom进行可视化的步骤：

安装和启动Visdom服务器：

```bash
pip install visdom
python -m visdom.server
```

在Python脚本中引入Visdom：

```python
import visdom
vis = visdom.Visdom()
```
定义模型和优化器：

```python
import torch
import torch.nn as nn
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
```

定义一个可视化损失函数的函数：

scss
```scss
def plot_loss(loss, epoch):
    vis.line(
        X=[epoch],
        Y=[loss],
        win='loss',
        update='append',
        opts=dict(title='Training Loss', xlabel='Epoch', ylabel='Loss')
    )
```
定义一个训练循环：

```python
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:
            plot_loss(running_loss / 10, epoch)
            running_loss = 0.0
```
创建一个散点图来可视化模型的输出：

```python
dataiter = iter(testloader)
images, labels = dataiter.next()
outputs = net(images)
vis.scatter(X=outputs.data, Y=labels)
```
这些是在PyTorch中使用Visdom进行可视化的一些基本步骤。您可以根据需要使用其他可用的函数和参数进行自定义。