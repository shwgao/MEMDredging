import torch
import torch.nn as nn
import copy  # Add this import at the top

# 固定随机种子保证可重复性
torch.manual_seed(42)

# 生成数据：batch_size=6，输入特征维度=10
X = torch.randn(6, 10)  # 输入数据
y = torch.tensor([[1.0], [0.0], [1.0], [0.0], [1.0], [0.0]])  # 目标值

# 定义简单线性模型，移除 Dropout 层
model = nn.Sequential(
    nn.Linear(10, 10),
    nn.BatchNorm1d(10),
    nn.ReLU(),
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.BatchNorm1d(10),
    nn.Linear(10, 1)
)

model2 = copy.deepcopy(model)  # Replace model.clone() with this

criterion = nn.MSELoss()  # 默认对 batch 取平均损失
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 前向传播
y_pred = model(X)
print(y_pred)
loss = criterion(y_pred, y)
print(loss)
# 反向传播自动计算梯度
# optimizer.zero_grad()
loss.backward()
# print(model[0](X))

# 获取自动计算的梯度（平均）
auto_grad_mean = model[0].weight.grad.clone()

# 手动计算平均梯度
# manual_grad_mean = torch.zeros_like(model2[0].weight.grad)
# optimizer.zero_grad()
inter_results = []
for i in range(len(X)//2):
    inter_i = model2[0](X[i*2:i*2+2])
    inter_i = model2[1](inter_i)
    inter_results.append(inter_i)
y_pred = torch.cat(inter_results, dim=0)
# print(y_pred)
for i in range(len(model2)-2):
    y_pred = model2[i+2](y_pred)
    
print(y_pred)
loss_i = criterion(y_pred, y)
print(loss_i)
loss_i.backward()
manual_grad_mean = model2[0].weight.grad.clone()

print("自动计算的梯度（平均）:\n", auto_grad_mean)
print("手动计算的梯度（平均）:\n", manual_grad_mean)