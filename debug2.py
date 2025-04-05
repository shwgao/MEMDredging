import torch
from rockmate import PureRockmate, Hiremate
from torchvision.models import resnet101

device = torch.device("cuda")

resnet = resnet101().cuda()
optimizer = torch.optim.Adam(resnet.parameters())
sample = torch.randn([100, 3, 128, 128]).cuda()
m_budget = 32 * 1024**3 # 2GB

use_hiremate = False

if use_hiremate:
	rk_resnet = Hiremate(resnet, sample, m_budget)
else:
	rk_resnet = PureRockmate(resnet, sample, m_budget)
 

print('--------------------------------')

dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(sample), batch_size=1, shuffle=True)

loss_function = torch.nn.MSELoss()

target = torch.randn([100, 1000]).cuda()

y = rk_resnet(sample) # use rk_resnet as resnet
loss = loss_function(y, target)
loss.backward()
optimizer.step() # parameters in resnet are updated