# 矩阵运算
```python

a = torch.randn(10000, 1000)
b = torch.randn(1000,2000)

device = torch.device("cuda")

a_c = a.to(device)
b_c = b.to(device)

c = torch.matmul(a_c, b_c)
```

# 求导


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

