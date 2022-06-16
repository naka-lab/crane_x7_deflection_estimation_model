from torch import nn, optim
import numpy as np
from matplotlib import pyplot as plt
import torch
import pandas as pd
 
# 線形回帰ネットワークのclassをnn.Moduleの継承で定義
class Estimate_defle1(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2,10)
        self.sigmoid = nn.Sigmoid()
        self.linear2 = nn.Linear(10,2)

    def forward(self, datass):
        datass = self.linear1(datass)
        datass = self.sigmoid(datass)
        datass = self.linear2(datass)

        return datass
    
    # きちんと学習できてるか確認用
    def test(self,z1,z2):
        z = torch.Tensor([z1,z2])
        out = net(z)/100
        return out

def test(z1,z2):
    z = torch.Tensor([z1,z2])
    out = net(z)/100
    return out

#データの読み込み
df = pd.read_csv(filepath_or_buffer="/home/nakalab/デスクトップ/ebara_crane/estimate_deflection/res0511.csv")

#指令値のz座標
data = torch.Tensor( [df.iloc[:,7],df.iloc[:,3]])
datas = torch.t(data)
datass = torch.tensor(datas, requires_grad=True)

#ARマーカーのz座標
label = torch.Tensor([df.iloc[:,8],df.iloc[:,6]])
labels = torch.t(label)
labelss = torch.tensor(labels, requires_grad=True)

# データ数
N = len(datass)

net = Estimate_defle1()
optimizer = torch.optim.Adam( net.parameters(), lr=0.001 )

predicts = net(datass)

loss_list = []
for i in range(20000):
    optimizer.zero_grad()
    predicts = net(datass)
    loss = torch.sqrt(1/N*torch.sum((labelss - predicts)**2))  
    loss.backward()
    optimizer.step()
    print(loss)
    loss_list.append( loss.detach().numpy())

plt.plot( range(len(loss_list)), loss_list ) 
plt.savefig("result.png")  

#モデルを保存
net = net.to('cpu')
torch.save(net.state_dict(), 'model_cpu_05111.pth')