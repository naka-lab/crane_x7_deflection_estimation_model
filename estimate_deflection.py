import torch
from torch import nn, optim
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
 
# 線形回帰ネットワークのclassをnn.Moduleの継承で定義
class DeflectionEstimator(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2,10)
        self.sigmoid = nn.Sigmoid()
        self.linear2 = nn.Linear(10,2)

    def forward(self, data):
        data = self.linear1(data)
        data = self.sigmoid(data)
        data = self.linear2(data)
        return data

#学習データの読み込み
df = pd.read_csv(filepath_or_buffer="train_data.csv")

#指令値（学習データ）のz座標と目標までの距離
data = torch.Tensor( [df.iloc[:,7],df.iloc[:,3]])
data = torch.t(data)
datas = torch.tensor(data, requires_grad=True)

#計測するARマーカー（ラベルデータ）のz座標と目標までの距離
label = torch.Tensor([df.iloc[:,8],df.iloc[:,6]])
label = torch.t(label)
labels = torch.tensor(label, requires_grad=True)

# データ数
N = len(datas)

net = DeflectionEstimator()
optimizer = torch.optim.Adam( net.parameters(), lr=0.001 )

#学習
loss_list = []
for i in range(20000):
    optimizer.zero_grad()
    predicts = net(datas)
    loss = torch.sqrt(1/N*torch.sum((labels - predicts)**2))  
    loss.backward()
    optimizer.step()
    print(loss)

#学習したモデルを保存
net = net.to('cpu')
torch.save(net.state_dict(), 'model.pth')