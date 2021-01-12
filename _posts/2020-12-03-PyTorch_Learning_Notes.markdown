---
layout: post
title: PyTorch Learning Notes
date: 2020-12-03
---

# 一、PyTorch 60min 入门教程

## 1. PyTorch 基础操作

### 1.1 创建张量

* 导包：

  `from __future__ import print_function`

* 构造 5×3 矩阵，不做初始化：

  `x = torch.empty(5, 3)`

* 构造随机初始化的矩阵：

  `x = torch.rand(5, 3)`

* 构造全 0 矩阵，数据类型全为 long：

  `x = torch.zeros(5, 3, dtype = torch.long)`

* 构造张量，直接使用数据：

  `x = torch.tensor([5.5, 3])`

* 创建一个 tensor 基于已经存在的 tensor：

  `x = x.new_ones(5, 3, dtype = torch.double)`

  `x = torch.randn_like(x, dtype = torch.float)`

* 获取 tensor 的维度信息

  `print(x.size())`

### 1.2 操作

* **加法**

  方式一：`print(x + y)`

  方式二：`print(torch.add(x, y))`

  ​                提供一个输出 tensor 作为参数：

  ```python
  result = torch.empty(5, 3)
  torch.add(x, y, out = result)
  print(result)
  ```

  方式三：`y.add_(x)`   `y.add_(1)`

  note : 任何使张量发生变化的操作都有一个前缀“_”

  `x.copy_(y), x.t_()(转置)`

* 切片索引：`print(x[:, 1])`

* 改变 tensor 的大小或形状

  ```python
  x = torch.randn(4, 4)
  y = x.view(16)
  z = x.view(-1, 8)    # -1处的维数由其他维度推断得到
  ```

* tensor 中只有一个元素，使用 .item() 来获得该 value

  ```python
  x = torch.randn(1)
  print(x.item())
  ```

* 将数组转化为 32bit 浮点型张量

  ```
  data = [-1,-2,1,2]
  tensor = torch.FloatTensor(data)
  ```
  
* 指数与对数、绝对值、求和

  ```python
  res = torch.exp(a)      # 底为e的指数(a是一个tensor)
  res = torch.log(a)      # 计算以e为底的对数
  res = torch.log2(a)     # 计算以2为底的对数
  res = torch.log10(a)    # 计算以10为底的对数
  res = torch.abs(a)      # 求绝对值
  res = torch.sum(a)      # 求和
  res = a.sum()
  ```

### 1.3 Torch 与 NumPy 的比较

* numpy 求均值：`np.mean(data)`

  tensor 求均值：`torch.mean(data)`

* numpy 矩阵相乘：`np.matmul(data, data)`    

  ```python
data = np.array(data)
  data.dot(data)
  ```
  
  tensor 矩阵相乘：`torch.mm(tensor, tensor)`

  `tensor.dot(tensor)`  （只支持一维数据）

### 1.4 NumPy Bridge

* 将 Torch Tensor 转化为 NumPy array

  ```python
  a = torch.ones(5)
  b = a.numpy()
  ```

* 将 NumPy Array 转化为 Torch Tensor

  ```python
  import numpy as np
  a = np.ones(5)
  b = torch.from_numpy(a)
  np.add(a, 1, out = a)
  ```

## 2. PyTorch 自动微分

* 创建一个张量，设置 requires_grad=True 来跟踪与它相关的计算

  ```python
  x = torch.ones(2, 2, requires_grad=True)    # 张量默认requires_grad是False
  y = x + 2
  print(y.grad_fn)
  z = y * y * 3
  out = z.mean()
  print(z, out)
  ```

  每个张量有一个 .grad_fn 属性保存着创建了张量的 Function 的引用

* .requires_grad_( ... ) 会改变张量的 requires_grad 标记（标记默认是False）

  ```python
  a = torch.randn(2, 2)
  a = ((a * 3) / (a - 1))
  print(a.requires_grad)
  a.requires_grad_(True)
  print(a.requires_grad)
  b = (a * a).sum()
  print(b.grad_fn)
  ```

* 后向传播

  ```python
  out.backward()    # 后向传播，等同于out.backward(torch.tensor(1.0))
  print(x.grad)     # 打印梯度d(out)/dx
  ```

* 想要雅可比向量积，只需简单的传递向量给 backward 作为参数

  ```python
  v = torch.tensor([0.1, 1.0, 0.0001],dtype=torch.float)
  y.backward(v)
  ```

* 通过将代码包裹在 with torch.no_grad() 里，停止对从跟踪历史中的 .requires_grad=True 的张量自动求导

  ```python
  print(x.requires_grad)
  print((x ** 2).requires_grad)
  
  with torch.no_grad():
      print((x ** 2).requires_grad)
  ```

* Variable 变量

  ```python
  tensor = torch.FloatTensor(data)
  variable = Variable(tensor, requires_grad = True)
  v_out = torch.mean(variable*variable)
  print(variable.grad)
  print(variable.data)    # tensor
  print(variable.data.numpy())    # numpy数组
  ```

## 3. PyTorch 神经网络

### 3.1 常见神经网络训练过程：

1. 定义一个包含可训练参数的神经网络
2. 迭代整个输入
3. 通过神经网络处理输入
4. 计算损失（loss）
5. 反向传播梯度到神经网路的参数
6. 更新网络的参数，典型简单的更新方法：weight = weight - learning_rate*gradient
7. 迭代训练
8. 调参、测试

（神经网络训练，一个batch更新一遍参数）

---

### 3.2 神经网络实现示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)
```

- 一个模型可训练的参数可以调用 net.parameters() 返回：

```python
params = list(net.parameters())
print(len(params))
print(params[0].size()) # conv1's .weight
```

- 输入和输出

```python
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)
```

- 把所有参数梯度缓存器置零，用随机梯度来反向传播

```python
net.zero_grad()
out.backward(torch.randn(1, 10))
```

- 损失函数：一个损失函数需要一对输入：模型输出和目标，然后计算一个值来评估输出距离目标有多远。 有一些不同的损失函数在 nn 包中。一个简单的损失函数就是 nn.MSELoss（还有 nn.CrossEntropyLoss），这计算了均方误差。 

```python
output = net(input)
target = torch.randn(10) # a dummy target, for example
target = target.view(1, -1) # make it the same shape as output

criterion = nn.MSELoss()
loss = criterion(output, target)    # loss = nn.MSELoss()(output, target)
print(loss)
```

- 反向传播步骤：

```python
print(loss.grad_fn) # MSELoss
print(loss.grad_fn.next_functions[0][0]) # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0]) # ReLU
```

​    为了实现反向传播损失，我们所有需要做的事情仅仅是使用 loss.backward()。你需要清空现存的
梯度，要不然将会和现存的梯度累计到一起。

```python
net.zero_grad() # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)
```

- 更新网络的参数：使用优化器（SGD、Adam、RMSProp等）

```python
import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad() # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step() # Does the update
```

- 迭代训练网络

```python
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
```

```python
for epoch in range(2):    # loop over the dataset multiple times
	running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %(epoch+1, i+1, running_loss/2000))
            running_loss = 0.0
            
print('Finished Training')
```

- 测试结果统计（测试集总的正确率）

```python
correct = 0
total = 0
with torch.no_grad():
	for data in testloader:
		images, labels = data
		outputs = net(images)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100*correct / total))
```

- 测试结果统计（每一类的正确率）

```python
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
	for data in testloader:
		images, labels = data
		outputs = net(images)
		_, predicted = torch.max(outputs, 1)
		c = (predicted == labels).squeeze()
		for i in range(4):
			label = labels[i]
			class_correct[label] += c[i].item()
			class_total[label] += 1
            
for i in range(10):
	print('Accuracy of %5s : %2d %%' %(classes[i], 100*class_correct[i]/class_total[i]))
```

- 在GPU上跑神经网络（把张量/神经网络转移到GPU上）

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

方法会递归地遍历所有模块，将它们的参数和缓冲器转换为 cuda 张量：`net.to(device)`

需要将数据也放在 GPU 上：`inputs, labels = inputs.to(device), labels.to(device)	`

*note :*  若网络比较小，则与 CPU 相比不会有很大的加速；data、tensor、model 都可以通过`.to(device)`放在 GPU 上。

## 4. PyTorch 之数据加载和处理

- 常用包：

```python
scikit-image    # 用于图像的io和变换
pandas          # 用于更容易地进行csv解析
```

- **数据集类：**`torch.utils.data.Dataset`是表示数据集的抽象类，自定义数据集应继承`Dataset`并覆盖以下方法：`__len__`实现`len(dataset)`返回数据集的尺寸；`__getitem__`用来获取一些索引数据，例如`dataset[i]`中的`i`。
- **数据加载：**对数据集简单使用`for`循环加载牺牲了许多性能，尤其是**批量处理数据**、**打乱数据**、**使用多线程`multiprocessingworker`并行加载数据**，`torch.utils.data.DataLoader`是一个提供上述功能的迭代器

```python
dataloader = DataLoader(XXX_dataset, batch_size=4, shuffle=True, num_workers=4)
# dataloader 返回的是 XXX_dataset 的返回内容，XXX_dataset通常为自行构造的Dataset的子类
```

# 二、PyTorch 小试牛刀

## 1. PyTorch核心：两个主要特征

- 一个n维张量，类似于numpy，但可以在GPU上运行
- 搭建和训练神经网络时的自动微分/求导机制

## 2. 张量

### 2.1 热身 :  Numpy

Numpy 实现简单神经网络

```python
# -*- coding: utf-8 -*-
import numpy as np

# N 批量大小，D_in 输入维度，H 隐藏维度，D_out 输出维度
N, D_in, H, D_out = 64, 1000, 100, 10

# 创建输入和输出随机数据
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# 随机初始化权重
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-6
for t in range(500):
    # 前向传递，计算预测值y
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)
    
    # 计算和打印损失loss
    loss = np.square(y_pred - y).sum()
    print(t, loss)
    
    # 反向传播，计算w1和w2对loss的梯度（不明白可手动推导）
    grad_y_pred = 2.0*(y_pred - y)           # loss对y_pred求导
    grad_w2 = h_relu.T.dot(grad_y_pred)      # loss对w2的梯度
    
    grad_h_relu = grad_y_pred.dot(w2.T)      # loss对h_relu求导
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)                # loss对w2的梯度
    
    # 更新权重
    w1 -= learning_rate*grad_w1
    w2 -= learning_rate*grad_w2
```

***note :*** Numpy 不能用 GPU 来加速数值计算。GPU 通常提供 50 倍或更高的加速。

### 2.2 PyTotch : 张量

**张量：**

PyTorch 的 tensor 在概念上与 numpy 的 array 相同：tensor 是一个n维数组，PyTorch 提供了许多函数用于操作这些张量。任何希望使用 NumPy 执行的计算也可以使用 PyTorch 的 tensor 来完成，可以认为它们是科学计算的通用工具。

与 Numpy 不同，PyTorch 可以利用 GPU 加速其数值计算。要在 GPU 上运行 Tensor，在构造张量时使用 device 参数把 tensor 建立在 GPU 上。

```python
# -*- coding: utf-8 -*-
import torch

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0")      # 取消注释以在GPU上运行

# N 批量大小，D_in 输入维度，H 隐藏维度，D_out 输出维度
N, D_in, H, D_out = 64, 1000, 100, 10

# 创建输入和输出随机数据
x = torch.randn(N, D_in, device = device, dtype = dtype)
y = torch.randn(N, D_out, device = device, dtype = dtype)

# 随机初始化权重
w1 = torch.randn(D_in, H, device = device, dtype = dtype)
w2 = torch.randn(H, D_out, device = device, dtype = dtype)

learning_rate = 1e-6
for t in range(500):
    # 前向传递，计算预测y
    h = x.mm(w1)
    h_relu = h.clamp(min=0)    # clamp 夹紧，该函数指定夹紧区间
    y_pred = h_relu.mm(w2)
    
    # 计算和打印损失
    loss = (y_pred - y).pow(2).sum().item()
    print(t, loss)
    
    # 反向传播计算w1和w2相对于损耗的梯度
    grad_y_pred = 2.0*(y_pred - y)           # loss对y_pred求导
    grad_w2 = h_relu.t().mm(grad_y_pred)     # loss对w2的梯度
    
    grad_h_relu = grad_y_pred.mm(w2.t())     # loss对h_relu求导
    grad_h = grad_h_relu.clone()             
    # .clone()返回一个张量的副本，其与原张量的尺寸和数据类型相同。
    # 与copy_()不同，这个函数记录在计算图中。传递到克隆张量的梯度将传播到原始张量。
    
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)                # loss对w2的梯度
    
    # 更新权重
    w1 -= learning_rate*grad_w1
    w2 -= learning_rate*grad_w2
    
```

### 2.3 自动求导

#### 2.3.1 张量和自动求导

- autograd 自动求导原理：手动实现反向传递对于小型双层网络来说并不是什么大问题，但对于大型复杂网络来说很快就会变得非常繁琐。但是可以使用自动微分来自动计算神经网络中的后向传递。 PyTorch中的autograd 包提供了这个功能。当使用autograd时，网络前向传播将定义一个计算图；图中的节点是tensor，边是函数，这些函数是输出tensor到输入tensor的映射。这张计算图使得在网络中反向传播时梯度的计算十分简单。 

- 实际使用：在建立tensor时加入`requires_grad = True`，这个tensor上的任何PyTorch的操作都将构造一个计算图，从而允许我们稍后在图中执行反向传播。反向传播之后 x.grad 将会是另一个张量，为x关于某个标量值的梯度。 可以使用`torch.no_grad()`上下文管理器来防止构造计算图。

```python
# -*- coding: utf-8 -*-
import torch

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0")      # 取消注释以在GPU上运行

# N 批量大小，D_in 输入维度，H 隐藏维度，D_out 输出维度
N, D_in, H, D_out = 64, 1000, 100, 10

# 创建输入和输出随机数据
# 设置 requires_grad = False 表示不需要追踪梯度
x = torch.randn(N, D_in, device = device, dtype = dtype)
y = torch.randn(N, D_out, device = device, dtype = dtype)

# 创建随机Tensor作为权重
# 设置 requires_grad = True 表示需要追踪梯度
w1 = torch.randn(D_in, H, device = device, dtype = dtype, requires_grad = True)
w2 = torch.randn(H, D_out, device = device, dtype = dtype, requires_grad = True)

learning_rate = 1e-6
for t in range(500):
    # 前向传播：使用tensors上的操作计算预测值y
    # 由于w1和w2有requires_grad=True，涉及这些张量的操作将让PyTorch构建计算图，
    # 从而允许自动计算梯度。由于我们不再手工实现反向传播，所以不需要保留中间值的引用。
    y_pred = x.mm(w1).clamp(min = 0).mm(w2)
    
    # 使用Tensors上的操作计算和打印丢失。
    # loss是一个形状为[1]的张量
    # loss.item()得到这个张量对应的python数值
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())
    
    # 使用autograd计算反向传播。这个调用将计算loss对所有requires_grad=True的tensor的梯度。
    # 这次调用后，w1.grad和w2.grad将分别是loss对w1和w2的梯度张量。
    loss.backward()
    
    # 使用梯度下降更新权重。对于这一步，我们只想对w1和w2的值进行原地改变；不想为更新阶段构建计算图，
	# 所以我们使用torch.no_grad()上下文管理器防止PyTorch为更新构建计算图
	with torch.no_grad():
		w1 -= learning_rate * w1.grad
		w2 -= learning_rate * w2.grad
        
	    # 反向传播后手动将梯度设置为零
	    w1.grad.zero_()
	    w2.grad.zero_()
```

​		***note :*** 更新权重时，需要停止梯度追踪。

#### 2.3.2 定义新的自动求导函数

- 在底层，每一个原始的自动求导运算实际上是两个在Tensor上运行的函数。其中，`forward`函数
  计算从输入Tensors获得的输出Tensors。而`backward`函数接收输出Tensors对于某个标量值的梯
  度，并且计算输入Tensors相对于该相同标量值的梯度。
- 在PyTorch中，我们可以很容易地通过定义 `torch.autograd.Function` 的子类并实现`forward` 和 `backward`函数，来定义自己的自动求导运算。之后我们就可以使用这个新的自动梯度运算符了。 

```python
import torch
class MyReLU(torch.autograd.Function):
    """
    我们可以通过建立torch.autograd的子类来实现我们自定义的autograd函数，并完成张量的正向和反向传播。
    """
    @staticmethod
    def forward(ctx, x):
        """
        在正向传播中，我们接收到一个上下文对象和一个包含输入的张量；
        我们必须返回一个包含输出的张量，
        并且我们可以使用上下文对象来缓存对象，以便在反向传播中使用。
        """
        ctx.save_for_backward(x)
        return x.clamp(min=0)
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        在反向传播中，我们接收到上下文对象和一个张量，
        其包含了相对于正向传播过程中产生的输出的损失的梯度。
        我们可以从上下文对象中检索缓存的数据，
        并且必须计算并返回与正向传播的输入相关的损失的梯度。
        """
        x, = ctx.saved_tensors
        grad_x = grad_output.clone()
        grad_x[x < 0] = 0
        return grad_x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# N 批量大小，D_in 输入维度，H 隐藏维度，D_out 输出维度
N, D_in, H, D_out = 64, 1000, 100, 10

# 创建输入和输出随机数据
# 设置 requires_grad = False 表示不需要追踪梯度
x = torch.randn(N, D_in, device = device, dtype = dtype)
y = torch.randn(N, D_out, device = device, dtype = dtype)

# 创建随机Tensor作为权重
# 设置 requires_grad = True 表示需要追踪梯度
w1 = torch.randn(D_in, H, device = device, dtype = dtype, requires_grad = True)
w2 = torch.randn(H, D_out, device = device, dtype = dtype, requires_grad = True)

learning_rate = 1e-6
for t in range(500):
    # 正向传播：使用张量上的操作来计算输出值y；
    # 我们通过调用MyReLU.apply函数来使用自定义的ReLU
    y_pred = MyReLU.apply(x.mm(w1)).mm(w2)
    
    # 计算并输出loss
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())
    
    # 使用autograd计算反向传播过程
    loss.backward()
    
    with torch.no_grad():
        # 用梯度下降更新权重
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad
        
        # 反向传播后手动将梯度设置为零
        w1.grad.zero_()
        w2.grad.zero_()
```

#### 2.3.3 Tensorflow : 静态图

- PyTorch自动求导看起来非常像TensorFlow：这两个框架中，我们都定义计算图，使用自动微分来
  计算梯度。两者最大的不同就是TensorFlow的计算图是静态的，而PyTorch使用动态的计算图。在TensorFlow中，我们定义计算图一次，然后重复执行这个相同的图，可能会提供不同的输入数
  据。而在PyTorch中，每一个前向通道定义一个新的计算图。 

```python
import tensorflow as tf
import numpy as np

# N 批量大小，D_in 输入维度，H 隐藏维度，D_out 输出维度
N, D_in, H, D_out = 64, 1000, 100, 10

# 为输入和目标数据创建placeholder；
# 当执行计算图时，他们将会被真实的数据填充
x = tf.placeholder(tf.float32, shape=(None, D_in))
y = tf.placeholder(tf.float32, shape=(None, D_out))

# 为权重创建Variable并用随机数据初始化
# TensorFlow的Variable在执行计算图时不会改变
w1 = tf.Variable(tf.random_normal((D_in, H)))
w2 = tf.Variable(tf.random_normal((H, D_out)))

# 前向传播：使用TensorFlow的张量运算计算预测值y。
# 注意这段代码实际上不执行任何数值运算；
# 它只是建立了我们稍后将执行的计算图。
h = tf.matmul(x, w1)
h_relu = tf.maximum(h, tf.zeros(1))
y_pred = tf.matmul(h_relu, w2)

# 使用TensorFlow的张量运算损失（loss）
loss = tf.reduce_sum((y - y_pred) ** 2.0)

# 计算loss对于w1和w2的导数
grad_w1, grad_w2 = tf.gradients(loss, [w1, w2])

# 使用梯度下降更新权重。为了实际更新权重，我们需要在执行计算图时计算new_w1和new_w2。
# 注意，在TensorFlow中，更新权重值的行为是计算图的一部分;
# 但在PyTorch中，这发生在计算图形之外。
learning_rate = 1e-6
new_w1 = w1.assign(w1 - learning_rate * grad_w1)
new_w2 = w2.assign(w2 - learning_rate * grad_w2)

# 现在我们搭建好了计算图，所以我们开始一个TensorFlow的会话（session）来实际执行计算图。
with tf.Session() as sess:
    
    # 运行一次计算图来初始化Variable w1和w2
    sess.run(tf.global_variables_initializer())
    
    # 创建numpy数组来存储输入x和目标y的实际数据
    x_value = np.random.randn(N, D_in)
    y_value = np.random.randn(N, D_out)
    
    for _ in range(500):
        # 多次运行计算图。每次执行时，我们都用feed_dict参数，
        # 将x_value绑定到x，将y_value绑定到y，
        # 每次执行图形时我们都要计算损失、new_w1和new_w2；
        # 这些张量的值以numpy数组的形式返回。
        loss_value, _, _ = sess.run([loss, new_w1, new_w2], feed_dict={x: x_value, y: y_value})
        print(loss_value)
```

### 2.4 nn模块

#### 2.4.1 PyTorch : nn

- 计算图和autograd是十分强大的工具，可以定义复杂的操作并自动求导；然而对于大规模的网
  络，autograd太过于底层。 在构建神经网络时，我们经常考虑将计算安排成层，其中一些具有可
  学习的参数，它们将在学习过程中进行优化。
- 在PyTorch中，包 nn 完成了同样的功能。nn包中定义一组大致等价于层的模块。一个模块接受输
  入的tesnor，计算输出的tensor，而且还保存了一些内部状态比如需要学习的tensor的参数等。nn
  包中也定义了一组损失函数（loss functions），用来训练神经网络。

使用`nn`模块实现两层神经网络：

```python
# -*- coding: utf-8 -*-
import torch

# N 批量大小，D_in 输入维度，H 隐藏维度，D_out 输出维度
N, D_in, H, D_out = 64, 1000, 100, 10

#创建输入和输出随机张量
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# 使用nn包将模型定义为一系列的层
# nn.Sequential 是包含其他模块的模块，并按顺序应用这些模块来产生其输出
# 每个线性模块使用线性函数从输入计算输出，并保存其内部的权重和偏差张量
# 在构造模型之后，使用.to()方法将其移动到所需的设备
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)

# nn包还包含常用的损失函数的定义；
# 在这种情况下，我们将使用平均平方误差(MSE)作为我们的损失函数。
# 设置reduction='sum'，表示我们计算的是平方误差的“和”，而不是平均值;
# 这是为了与前面我们手工计算损失的例子保持一致，
# 但是在实践中，通过设置reduction='elementwise_mean'来使用均方误差作为损失更为常见。
loss_fn = torch.nn.MSELoss(reduction = 'sum')

learning_rate = 1e-4
for t in range(500):
    # 前向传播：通过向模型传入x计算预测y
    # 模块对象重载了__call__运算符，所有可以像函数那样调用它们
    #（即实现了__call__方法的类可以通过类名直接调用而不用实例化）
    # 这么做相当于向模块传入了一个张量，然后它返回了一个输出张量
    y_pred = model(x)
    
    # 计算并打印损失
    # 传递包含y的预测值和真实值的张量，损失函数返回包含损失的张量
    loss = loss_fn(y_pred, y)
    print(t, loss.item())
    
    # 反向传播之前清零梯度（问题：到底是反向传播之前还是之后清零？？）
    model.zero_grad()
    
    # 反向传播：计算模型的损失对所有可学习参数的导数（梯度）。
    # 在内部，每个模块的参数存储在requires_grad=True的张量中，
    # 因此这个调用将计算模型中所有可学习参数的梯度。
    loss.backward()
    
    # 使用梯度下降更新权重
    # 每个参数都是张量，所以可以像以前那样得到它的数值和梯度
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate*param.grad      # 改变量为：学习率*梯度

# 小疑问：nn模块实现的网络loss比手动实现的网络loss小很多个数量级
    
```

#### 2.4.2 PyTorch : optim

- 优化器/优化算法：SGD、AdaGrad、RMSProp、Adam

```python
import torch

# N 批量大小，D_in 输入维度，H 隐藏维度，D_out 输出维度
N, D_in, H, D_out = 64, 1000, 100, 10

#创建输入和输出随机张量
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# 使用nn包定义模型和损失函数
model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
)
loss_fn = torch.nn.MSELoss(reduction = 'sum')

# 使用optim包定义优化器（Optimizer）。Optimizer将会为我们更新模型的权重。
# 这里我们使用Adam优化方法；optim包还包含了许多别的优化算法。
# Adam构造函数的第一个参数告诉优化器应该更新哪些张量。
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

for t in range(500):
    
    # 前向传播：通过向模型输入x计算预测的y
    y_pred = model(x)
    
    # 计算并打印loss
    loss = loss_fn(y_pred, y)
    print(t, loss.item())
    
    # 在反向传播之前，使用optimizer将它要更新的所有张量的梯度清零(这些张量是模型可学习的权重)
    optimizer.zero_grad()
    
    # 反向传播：根据loss对于模型的参数的梯度
    loss.backward()
    
    # 调用Optimizer的step函数使它所有参数更新
    optimizer.step()
```

#### 2.4.3 自定义 nn 模块

- 有时候需要指定比现有模块序列更复杂的模型，对于这些情况，可以通过继承 nn.Module 并定义 forward 函数，这个 forward 函数可以使用其他模块或者其他的自动求导运算来接收输入 tensor，产生输出tensor。 

用自定义的`Module`子类构建两层神经网络：

```python
import torch

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        '''
        在构造函数中，实例化两个nn.Linear模块，并将它们作为成员变量
        '''
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)
    
    def forward(self, x):
        """
        在前向传播的函数中，我们接收一个输入的张量，也必须返回一个输出张量。
        我们可以使用构造函数中定义的模块以及张量上的任意的（可微分的）操作。
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred
    
# N 批量大小，D_in 输入维度，H 隐藏维度，D_out 输出维度
N, D_in, H, D_out = 64, 1000, 100, 10

# 创建输入和输出随机张量
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# 通过实例化上面定义的类来构建模型
model = TwoLayerNet(D_in, H, D_out)

# 构造损失函数和优化器
# SGD构造函数中对model.parameters()的调用，
# 将包含模型的一部分，即两个nn.Linear模块的可学习参数。
loss_fn = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-4)

for t in range(500):
    # 前向传播：通过向模型传递x计算预测值y
    y_pred = model(x)
    
    # 计算并输出loss
    loss = loss_fn(y_pred, y)
    print(t, loss.item())
    
    # 清零梯度，反向传播，更新权重
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

#### 2.4.4 控制流和权重共享

- 动态图和权重共享示例：一个全连接的ReLU网络，在每一次前向传播时，它的隐藏层的层数为随机1到4之间的数，这样可以多次重用相同的权重来计算。 

```python
import random
import torch

class DynamicNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        在构造函数中，我们构造了三个nn.Linear实例，它们将在前向传播时被使用。
        """
        super(DynamicNet, self).__init__()
        self.input_linear = torch.nn.Linear(D_in, H)
        self.middle_linear = torch.nn.Linear(H, H)
        self.output_linear = torch.nn.Linear(H, D_out)
        
    def forward(self, x):
        """
        对于模型的前向传播，我们随机选择0、1、2、3，并重用了多次计算隐藏层的middle_linear模块。
        由于每个前向传播构建一个动态计算图，
        我们可以在定义模型的前向传播时使用常规Python控制流运算符，如循环或条件语句。
        在这里，我们还看到，在定义计算图形时多次重用同一个模块是完全安全的。
        这是对Lua Torch的一大改进，因为Lua Torch中每个模块只能使用一次。
        """
        h_relu = self.input_linear(x).clamp(min=0)
        for _ in range(random.randint(0, 3)):
            h_relu = self.middle_linear(h_relu).clamp(min=0)
        y_pred = self.output_linear(h_relu)
        return y_pred
    
# N 批量大小，D_in 输入维度，H 隐藏维度，D_out 输出维度
N, D_in, H, D_out = 64, 1000, 100, 10

# 创建输入和输出随机张量
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# 实例化上面的类来构造我们的模型
model = DynamicNet(D_in, H, D_out)

# 构造我们的损失函数（loss function）和优化器（Optimizer）。
# 用平凡的随机梯度下降训练这个奇怪的模型是困难的，所以我们使用了momentum方法
loss_fn = torch.nn.MSELoss(reduction = 'sum')
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-4, momentum = 0.9)

for t in range(500):
    
    # 前向传播，通过向模型传入x计算预测的y
    y_pred = model(x)
    
    # 计算并打印损失
    loss = loss_fn(y_pred, y)
    print(t, loss.item())
    
    # 清零梯度，反向传播，更新参数
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## 3. 保存和加载模型

- 当保存和加载模型时，需要熟悉三个核心功能：

1. `torch.save`  :  将序列化对象保存到磁盘。此函数使用Python的 pickle 模块进行序列化。使
   用此函数可以保存如模型、tensor、字典等各种对象。
2. `torch.load`  :  使用pickle的 unpickling 功能将pickle对象文件反序列化到内存。此功能还可
   以有助于设备加载数据。
3. `torch.nn.Module.load_state_dict`  :  使用反序列化函数 state_dict 来加载模型的参数字典。 

---

### 3.1 状态字典 state_dict

- 在 PyTorch 中，`torch.nn.Module` 模型的可学习参数（即权重和偏差）包含在模型的参数中，（使用 `model.parameters()` 可以进行访问）。 
- `state_dict` 是 Python 字典对象，它将每一层映射到其参数张量。注意，只有具有可学习参数的层（如卷积层，线性层等）的模型才具有 `state_dict` 这一项。
- 目标优化 `torch.optim` 也有 `state_dict` 属性，它包含有关优化器的状态信息，以及使用的超参数。
-  因为`state_dict`的对象是 Python 字典，所以它们可以很容易的保存、更新、修改和恢复，为 PyTorch 模型和优化器添加了大量模块。

简单分类器模型了解 `state_dict` 的使用：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义模型
class TheModelClass(nn.Module):
    
    def __init__(self):
        super(TheModelClass, self).__init__()
        # nn.Conv2d stride 默认为1; kernel_size为5即(5,5)的方形kernel，若非方形则(a, b)
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 5)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)    # 重新调整 tensor 的形状
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)           # 最后就不加激活函数了
        return x
    
# 初始化模型
model = TheModelClass()
    
# 初始化优化器
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)
    
# 打印模型的状态字典
print("model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        
# 打印优化器的状态字典
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])
```

### 3.2 保存和加载推理模型

#### 3.2.1 保存/加载 state_dict（推荐使用）

- 保存：`torch.save(model.state_dict(), PATH)`
- 加载：

```python
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()
```

- 当保存好模型用来推断的时候，只需要保存模型学习到的参数，使用 `torch.save()` 函数来保存模型的 `state_dict` ，它会给模型恢复提供 最大的灵活性，这就是为什么要推荐它来保存的原因。 

- 模型文件扩展名：`.pt` 或 `.pth`
- 在运行推理之前，务必调用 `model.eval()` 去设置 dropout 和 batch normalization 层为评估模式。如果不这么做，可能导致模型推断结果不一致。

***note :***  `load_state_dict()` 函数只接受字典对象，而不是保存对象的路径。这就意味着在你传给`load_state_dict()` 函数之前，你必须反序列化你保存的 `state_dict` 。例如，你无法通过 `model.load_state_dict(PATH)` 来加载模型。 

#### 3.2.2 保存/加载完整模型

- 保存：`torch.save(model, PATH)`
- 加载：

```python
# 模型类必须在此之前被定义
model = torch.load(PATH)
model.eval()
```

- 此部分保存/加载过程使用最直观的语法并涉及最少量的代码。以 Python pickle 模块的方式来保存模型。
- 这种方法的缺点是序列化数据受限于某种特殊的类而且需要确切的字典结构。这是因为 pickle 无法保存模型类本身。

### 3.3 保存和加载 CheckPoint 用于推理/继续训练

- 保存：

```python
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    ...
}, PATH)
```

- 加载：

```python
model = TheModelClass(*args, **kwargs)
optimizer = TheOptimizerClass(*args, **kwargs)

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
# - or -
model.train()
```

- 要保存多个组件，请在字典中组织它们并使用 `torch.save()` 来序列化字典。PyTorch 中保存 checkpoint 常用 `.tar` 文件扩展名。
- 要加载项目，首先需要初始化模型和优化器，然后使用 `torch.load()` 来加载本地字典。这里，你可以非常容易的通过简单查询字典来访问你所保存的项目。
- 在运行推理之前，务必调用 `model.eval()` 去设置 dropout 和 batch normalization 为评估。如果不这样做，有可能得到不一致的推断结果。
- 如果你想要恢复训练，请调用 `model.train()` 以确保这些层处于训练模式。

### 3.4 在一个文件中保存多个模型

- 保存：

```python
torch.save({
    'modelA_state_dict': modelA.state_dict(),
    'modelB_state_dict': modelB.state_dict(),
    'optimizerA_state_dict': optimizerA.state_dict(),
    'optimizerB_state_dict': optimizerB.state_dict(),
    ...
}, PATH)
```

- 加载：

```python
modelA = TheModelAClass(*args, **kwargs)
modelB = TheModelBClass(*args, **kwargs)
optimizerA = TheOptimizerAClass(*args, **kwargs)
optimizerB = TheOptimizerBClass(*args, **kwargs)

checkpoint = torch.load(PATH)
modelA.load_state_dict(checkpoint['modelA_state_dict'])
modelB.load_state_dict(checkpoint['modelB_state_dict'])
optimizerA.load_state_dict(checkpoint['optimizerA_state_dict'])
optimizerB.load_state_dict(checkpoint['optimizerB_state_dict'])

modelA.eval()
modelB.eval()
# - or -
modelA.train()
modelB.train()
```

### 3.5 使用在不同模型参数下的热启动模式

- 保存：`torch.save(modelA.state_dict(), PATH) `
- 加载：

```python
modelB = TheModelBClass(*args, **kwargs)
modelB.load_state_dict(torch.load(PATH), strict = False)
```

- 在迁移学习或训练新的复杂模型时，部分加载模型或加载部分模型是常见的情况。利用训练好的参数，有助于热启动训练过程，并希望帮助你的模型比从头开始训练能够更快地收敛。
- 无论是从缺少某些键的 `state_dict` 加载还是从键的数目多于加载模型的 `state_dict` ，都可以通过在
  `load_state_dict()` 函数中将 strict 参数设置为 `False` 来忽略非匹配键的函数。
- 如果要将参数从一个层加载到另一个层，但是某些键不匹配，主要修改正在加载的 `state_dict` 中的参数键的名称以匹配要在加载到模型中的键即可。

### 3.6 通过设备保存/加载模型

#### 3.6.1 保存到 CPU、加载到 CPU

- 保存：`torch.save(model.state_dict(), PATH) `
- 加载：

```python
device = torch.device('cpu')
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location=device))
```

#### 3.6.2 保存到 GPU、加载到 GPU

- 保存：`torch.save(model.state_dict(), PATH)`
- 加载：

```python
device = torch.device("cuda")
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.to(device)
# 确保在你提供给模型的任何输入张量上调用input = input.to(device)
```

***note :*** 调用 `my_tensor.to(device)` 会在GPU上返回 my_tensor 的副本。因此，请记住手动覆盖张量：`my_tensor = my_tensor.to(torch.device('cuda'))`。

#### 3.6.3 保存到 CPU、加载到GPU

- 保存：`torch.save(model.state_dict(), PATH) `
- 加载：

```python
device = torch.device("cuda")
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location="cuda:0")) # Choose whatever GPU device number you want
model.to(device)
# 确保在你提供给模型的任何输入张量上调用 input = input.to(device)
```

#### 3.6.4 保存 torch.nn.DataParallel 模型

- 保存：`torch.save(model.module.state_dict(), PATH) `
- 加载：`torch.nn.DataParallel` 是一个模型封装，支持并行GPU使用。要普通保存 `DataParallel` 模型，请保存 `model.module.state_dict()`。这样，你就可以非常灵活地以任何方式加载模型到你想要的设备中。

# 三、Pytorch 简介

## 1. Torch 张量库介绍

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
```

### 1.1 创建张量

```python
# 利用给定数据创建一个 torch.Tensor 对象，是一个一维向量
V_data = [1., 2., 3.]
V = torch.Tensor(V_data)
print(V)

# 创建一个矩阵
M_data = [[1., 2., 3.], [4., 5., 6.]]
M = torch.Tensor(M_data)
print(M)

# 创建 2x2x2 形式的三维张量
T_data = [[[1., 2.,], [3., 4.]], [[5., 6.,], [7., 8.]]]
T = torch.Tensor(T_data)
print(T)
```

- 什么是三维张量？

​    对向量索引，得到标量；对矩阵索引，得到向量；对三维张量索引，得到矩阵。

​    标量是零维张量，向量是一维张量，矩阵是二维张量。

```python
# 索引V得到一个标量（0维张量）
print(V[0])

# 从向量V中获取一个数字
print(V[0].item())

# 索引M得到一个向量
print(M[0])

# 索引T得到一个矩阵
print(T[0])
```

- 创建整数类型张量：`torch.LongTensor()`
- 随机数据张量：`torch.randn()`

```python
x = torch.randn((3, 4, 5))
print(x)
```

### 1.2 张量操作

- 相加

```python
x = torch.Tensor([1., 2., 3.])
y = torch.Tensor([4., 5., 6.])
z = x + y
print(z)
```

- 连接

```python
# 默认情况下，连接行
x_1 = torch.randn(2, 5)
y_1 = torch.randn(3, 5)
z_1 = torch.cat([x_1, y_1])
print(z_1)

# 连接列
x_2 = torch.randn(2, 3)
y_2 = torch.randn(2, 5)
z_2 = torch.cat([x_2, y_2], 1)    # 第二个参数指定了沿着哪条轴连接
print(z_2)

# 如果tensors是不兼容的，torch会报错
# torch.cat([x_1, x_2])
```

### 1.3 重构张量

```python
# 使用 .view() 重构张量
x = torch.randn(2, 3, 4)
print(x)
print(x.view(2, 12))
print(x)
print(x.view(2, -1))    # 如果设置维度为-1，其大小可以根据数据被推断出来
```

## 2. 计算图和自动求导

如果 `requires_grad=True `，张量对象可以一直跟踪它是如何创建的。

```python
# 张量对象带有“requires_grad”标记
x = torch.tensor([1., 2., 3], requires_grad=True)
# 通过requires_grad=True，您也可以做之前所有的操作。
y = torch.tensor([4., 5., 6], requires_grad=True)
z = x + y
print(z)
# 但是z还有一些额外的东西.
print(z.grad_fn)
```

***note :***  `torch.Tensor` 是一个大类，利用列表构建 tensor 要用小写的 `torch.tensor`

如果你继续跟踪 `z.grad_fn` ，你会从中 找到x和y的痕迹。

```python
# 我们来将z中所有项作和运算
s = z.sum()
print(s)
print(s.grad_fn)
```

Pytorch 程序的开发人员用程序指令 `sum()` 和 + 操作以知道如何计算它们的梯度并且运行反向传播算法。

Pytorch累积梯度渐变为 `.grad` 属性。

```python
# 在任意变量上使用.backward()将会运行反向，从它开始
s.backward()
print(x.grad)
```

---

- 关于自动微分与分离

```python
x = torch.randn((2, 2))
y = torch.randn((2, 2))
# 用户创建的张量在默认情况下，requires_grad = False
print(x.requires_grad, y.requires_grad)
z = x + y
# 不能通过z反向传播
print(z.grad_fn)

# .requires_grad_()改变了 requires_grad 属性
# 如果没有指定，标记默认为True；若要指定，x = x.requires_grad_(False)
x = x.requires_grad_()
y = y.requires_grad_()
print(x.requires_grad, y.requires_grad)
# z包含足够的信息计算梯度
z = x + y
print(z.grad_fn)
# 如果任何操作的输入部分带有“requires_grad=True”那么输出就会变为：
print(z.requires_grad)

# 现在z有关于x,y的历史信息
# 我们可以获取它的值，将其从历史中分离出来吗？
new_z = z.detach()

# new_z 有足够的信息反向传播至x和y吗？
# 答案是没有
print(new_z.grad_fn)
# 怎么会这样？“z.detach()”函数返回了一个与“z”相同存储的张量
# 但是没有携带历史的计算信息。
# 它对于自己是如何计算得来的不知道任何事情。
# 从本质上讲，我们已经把这个变量从过去的历史中分离出来了。
```

也可以通过 `with torch.no_grad()` 停止跟踪张量的历史记录中的自动求导：

```python
print(x.requires_grad)
print((x**2).requires_grad)
with torch.no_grad():
    print((x**2).requires_grad)
```

## 3. 使用 PyTorch 进行深度学习

### 3.1 使用深度学习构建模块：仿射变换、非线性函数以及目标函数

#### 3.1.1 仿射变换

- 仿射变换：`f(x) = Ax +b`

```python
import torch
import torch.nn as nn

torch.manual_seed(1)

# PyTorch 的映射输入是行而不是列，输出的第i行是对输入的第i行进行A变换并加上偏移项
data = torch.randn(2, 5)
lin = nn.Linear(5, 3)
print(lin(data))    # 输出是2行3列
```

#### 3.1.2 非线性函数

​    使用以上方法将多个仿射变换组合成的长链式的神经网络，相对于单个仿射变换并没有性能上的提升。但是如果我们在两个仿射变换之间引入非线性，那么结果就大不一样了，我们可以构建出一个高性能的模型。最常用的核心的非线性函数有 `tanh(x), σ(x), ReLU(x)`，这些函数拥有可以计算的梯度，而计算梯度是学习的本质。例如，`dσ/dx = σ(x)(1-σ(x))`。

***note :***  对于`σ(x)`，当参数的绝对值增长时，梯度会很快消失，小梯度意味着很难学习。因此大部分人默认选择 `tanh` 或者 `ReLU`。

```python
import torch
import torch.nn.functional as F
# 在PyTorch中，大多数非线性函数都在torch.nn.functional中，把它导入为F
# 注意，非线性函数通常没有向仿射变换那样的参数
# 也就是说，它们没有在训练期间更新的权重
data = torch.randn(2, 2)
print(data)
print(F.relu(data))
```

#### 3.1.3 Softmax 和概率

​    `Softmax(x)`也是一个非线性函数，但它的特殊之处在于，它通常是神经网络的最后一个操作。这是因为它接受实数向量，并且返回一个概率分布，每个元素都非负且和为1。

​    `softmax(x)` 的第 i 个分量是：`exp(xi)/Σj exp(xj)`。

​    也可以认为这只是一个对输入的元素进行的求幂运算符，使所有的内容都非负，然后除以规范化常量。

```python
# softmax也在F中
data = torch.randn(5)
print(data)
print(F.softmax(data, dim = 0))
print(F.softmax(data, dim = 0).sum())    # 总和为1，因为它是一个分布
print(F.log_softmax(data, dim = 0))      # 也有 log_softmax
```

- log_softmax 是在 softmax 的结果上多做一次 log 运算（PyTorch 实现中使用 ln）

#### 3.1.4 目标函数

​    **目标函数**正是神经网络通过训练来最小化的函数（因此，它常常被称作**损失函数**或者**成本函数**）。这需要首先选择一个训练数据实例，通过神 经网络运行它并计算输出的损失。然后通过损失函数的导数来更新模型的参数。

### 3.2 优化和训练

- 参数更新：`θ(t+1) = θ(t) - η ▽θ L(θ)`
- 你不需要担心这些特殊的算法到底在干什么，除非你真的很感兴趣。Torch提供了大量的算法在torch.optim包中，且全部都是透明的。在语法 上使用复杂的算法和使用最简单的梯度更新一样简单。
- 但是尝试不同的更新算法和在更新算法中使用不同的参数（例如不同的初始学习率）对 于优化你的网络的性能很重要。通常，仅仅将普通的 *SGD* 替换成一个例如 *Adam* 或者 *RMSProp* 优化器都可以显著提升性能。

### 3.3 使用 PyTorch 创建网络组件

​    所有的网络组件应该继承 `nn.Module` 并覆盖 `forward()` 方法。继承 `nn.Module` 提供给了一些方法给你的组件。例如，它可以跟踪可训练的参数，你可以通过 `.to(device)` 方法在 CPU 和 GPU 之间交换它们，其中的 device 可以是 CPU 设备 `torch.device("cpu")` 或者 CUDA 设备 `torch.device("cuda:0")`。

​    让我们写一个神经网络的示例，它接受一些稀疏的 BOW (词袋模式) 表示，然后输出分布在两个标签上的概率：“English” 和 “Spanish”。这个模型只是一个逻辑回归。

---

#### 示例：基于逻辑回归与词袋模式的文本分类器

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

data = [("me gusta comer en la cafeteria".split(), "SPANISH"),
        ("Give it to me".split(), "ENGLISH"),
        ("No creo que sea una buena idea".split(), "SPANISH"),
        ("No it is not a good idea to get lost at sea".split(), "ENGLISH")]
test_data = [("Yo creo que si".split(), "SPANISH"),
             ("it is lost on me".split(), "ENGLISH")]

word_to_ix = {}
for sent, _ in data + test_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print(word_to_ix)

VOCAB_SIZE = len(word_to_ix)
NUM_LABELS = 2

class BoWClassifier(nn.Module):
    def __init__(self, num_labels, vocab_size):
        super(BoWClassifier, self).__init__()
        self.linear = nn.Linear(vocab_size, num_labels)
        
    def forward(self, bow_vec):
        return F.log_softmax(self.linear(bow_vec), dim=1)
    
# 计数词频，作为向量（BOW原理）
def make_bow_vector(sentence, word_to_ix):
    vec = torch.zeros(len(word_to_ix))
    for word in sentence:
        vec[word_to_ix[word]] += 1
    return vec.view(1, -1)
    
def make_target(label, label_to_ix):
    return torch.LongTensor([label_to_ix[label]])
    
model = BoWClassifier(NUM_LABELS, VOCAB_SIZE)

# 模型知道它的参数。 下面的第一个输出是A，第二个输出是b。
# 无论何时将组件分配给模块的__init__函数中的类变量，都是使用self.linear = nn.Linear(...)行完成的。
# 然后通过PyTorch，你的模块（在本例中为BoWClassifier）将存储nn.Linear参数的知识
for param in model.parameters():
    print(param)
    
# 要运行模型，请传入BoW矢量
# 这里我们不需要训练，所以代码包含在torch.no_grad（）中
with torch.no_grad():
    sample = data[0]
    bow_vector = make_bow_vector(sample[0], word_to_ix)
    log_probs = model(bow_vector)
    print(log_probs)
    
label_to_ix = {"SPANISH": 0, "ENGLISH": 1}

```

**训练：**将实例传入来获取对数概率，计算损失函数，计算损失函数的梯度，然后使用一个梯度步长来更新参数。在 PyTorch 的 `nn` 包里提供了损失函数。`nn.NLLLoss()` 是我们想要的负对数似然损失函数。优化方法使用SGD。

​    因为 `NLLLoss()` 的输入是一个对数概率的向量以及目标标签。它不会为我们计算对数概率。这也是为什么我们最后一层网络是 `log_softmax` 的原因。损失函数 `nn.CrossEntropyLoss()` 除了对结果额外计算了`logsoftmax`之外，和`NLLLoss()` 没什么区别。

```python
# 在我们训练之前运行测试数据，只是为了看到之前-之后
with torch.no_grad():
    for instance, label in test_data:
        bow_vec = make_bow_vector(instance, word_to_ix)
        log_probs = model(bow_vec)
        print(log_probs)
        
# 打印与“creo”对应的矩阵列
print(next(model.parameters())[:, word_to_ix["creo"]])

loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 通常，您希望多次传递训练数据.
# 100比实际数据集大得多，但真实数据集有两个以上的实例。
# 通常，在5到30个epochs之间是合理的。
for epoch in range(100):
    for instance, label in data:
        # 步骤1： 请记住，PyTorch会累积梯度。所以每个样本训练前要清零
        model.zero_grad()
        
        # 步骤2：制作我们的BOW向量，并且我们必须将目标作为整数包装在Tensor中。
        # 例如，如果目标是SPANISH，那么我们包装整数0
        # 然后，loss函数知道对数概率的第0个元素是对应于SPANISH的对数概率
        bow_vec = make_bow_vector(instance, word_to_ix)
        target = make_target(label, label_to_ix)
        
        # 步骤3：运行我们的前向传递
        log_probs = model(bow_vec)
        
        # 步骤4： 通过调用optimizer.step()来计算损失，梯度和更新参数
        loss = loss_function(log_probs, target)
        print(epoch, loss.item())
        loss.backward()
        optimizer.step()
        
with torch.no_grad():
    for instance, label in test_data:
        bow_vec = make_bow_vector(instance, word_to_ix)
        log_probs = model(bow_vec)
        print(log_probs)
        
# 对应西班牙语的指数上升，英语下降！
print(next(model.parameters())[:, word_to_ix["creo"]])
```

# 四、PyTorch 其他

## 1. nn.Embedding 学习

- nn.Embedding 可以理解为一个词典的 embedding，从中可以索引到某个单词的 embedding。

```python
import torch
import torch.nn as nn

emb = nn.Embedding(10, 3)
print(emb)
print(emb.weight)         # emb.weight 的类型是 torch.nn.parameter.Parameter
print(emb.weight.data)    # 查看词典embedding;  emb.weight.data的类型是tensor

# 将三元组中的头尾实体和关系都映射为 embedding
inputTriples = torch.LongTensor([[2,5,2],[0,5,7],[1,6,9],[8,4,3]])
head, relation, tail = torch.chunk(inputTriples, chunks = 3, dim = 1)
print(inputTriples)
print(head)
print(relation)
print(tail)
print(head.size())

heads = torch.squeeze(head)    # 将头实体拍扁（即将维度为1的维度去掉，从竖条矩阵变成一个向量）
head_emb = emb(heads)          # 将每个头实体由标号转为embedding
print(head_emb)
```

