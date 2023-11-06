
# 工具
wisdom 可视化工具
np.meshgrdi

# 数据
## DataSet
```python
class MyDataSet(Dataset):
    def __init__(self):
        self.sample_list = ...
 
    def __getitem__(self, index):
        x= ...
        y= ...
        return x, y
 
    def __len__(self):
        return len(self.sample_list)
```
## DataLoader
```python
from torch.utils.data import DataLoader
# batch_size=4表示每次取四个数据
# shuffle= True表示开启数据集随机重排，即每次取完数据之后，打乱剩余数据的顺序，然后再进行下一次取
# num_workers=0表示在主进程中加载数据而不使用任何额外的子进程，如果大于0，表示开启多个进程，进程越多，处理数据的速度越快，但是会使电脑性能下降，占用更多的内存
# drop_last=False表示不丢弃最后一个批次，假设我数据集有10个数据，我的batch_size=3，即每次取三个数据，那么我最后一次只有一个数据能取，如果设置为true，则不丢弃这个包含1个数据的子集数据，反之则丢弃
test_load = DataLoader(dataset=test_data, batch_size=4 , shuffle= True, num_workers=0,drop_last=False)
```


# 函数

torch.clamp_min_方法设置一个下限min，tensor中有元素小于这个值, 就把对应的值赋为min
```python
x = torch.tensor([[1., 2. ,3. ,4., 5.]])
x = x.clamp_min(3)  #[3,3,3,4,5]
x.mean() #平均数3
```


# 矩阵运算


# 求导
```python
x = torch.randn(10, 3)  #输入，10个数据，3个特征
y = torch.randn(10, 2)  #输出，10个数据，2个

# Build a fully connected layer.
linear = nn.Linear(3, 2) #输入3个特征，输出2维
print ('w: ', linear.weight) # w是2*3得矩阵（偏置），因为一个1*3的输出，乘以3*2的矩阵，才能得到一个1*2的结果
print ('b: ', linear.bias) #偏置是1*2，跟结果保持一致

# Build loss function and optimizer.
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

# Forward pass.
pred = linear(x) # 算预测值

# Compute loss.
loss = criterion(pred, y) #计算损失
print('loss: ', loss.item())

# Backward pass.
loss.backward() 

# Print out the gradients.
print ('dL/dw: ', linear.weight.grad) 
print ('dL/db: ', linear.bias.grad)
```

# 模型

```python 
def __init__(self, in_features: int, out_features: int, bias: bool = True,  
device=None, dtype=None) -> None:
```

简单线性模型
```python
import torch  
  
#数据集  
x_data = torch.tensor([[1.0], [2.0], [3.0]])  
y_data = torch.tensor([[2.0], [4.0], [6.0]])  
  
# 定义模型  
class LinearModel(torch.nn.Module):  
	def __init__(self):  
		super(LinearModel, self).__init__()  
		self.linear = torch.nn.Linear(1, 1)  
  
	def forward(self, x):  
		y_pred = self.linear(x)  
		return y_pred  
  
model = LinearModel()  
  
# 定义损失函数  
criterion = torch.nn.MSELoss(size_average = False)  
# 优化器  
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # model.parameters()自动完成参数的初始化操作，这个地方我可能理解错了  
  
# 训练100轮  
for epoch in range(100):  
	y_pred = model(x_data) # forward:predict  
	loss = criterion(y_pred, y_data) # forward: loss  
	print(epoch, loss.item())  
  
	optimizer.zero_grad() # 非常重要  
	loss.backward() # backward: autograd，自动计算梯度  
	optimizer.step() # update 参数，即更新w和b的值  
  
# 打印训练出的参数  
print('w = ', model.linear.weight.item())  
print('b = ', model.linear.bias.item())  
  
#测试数据  
x_test = torch.tensor([[4.0]])  
print('y_pred = ', model(x_test))
```



# 项目

情感分析：https://github.com/bentrevett/pytorch-sentiment-analysis

