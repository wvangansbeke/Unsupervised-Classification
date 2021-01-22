import torch 
import torch.nn as nn

class Cell_Net(nn.Module):
    def __init__(self,in_channel=3):
        super(Cell_Net,self).__init__()
        #layer 1
        self.conv1=nn.Conv2d(in_channels=in_channel,out_channels=32, kernel_size=11,padding=1, stride=2,bias=False)
        self.maxpool1=nn.MaxPool2d(kernel_size=2,stride=2)
        self.act1=nn.ReLU()
        self.batchnorm1=nn.BatchNorm2d(32)

        #layer 2
        self.conv2=nn.Conv2d(in_channels=32,out_channels=64, kernel_size=6, stride=2,bias=False)
        self.maxpool2=nn.MaxPool2d(kernel_size=2,stride=2) #poolsize=2 in tf
        self.act2=nn.ReLU()
        self.batchnorm2=nn.BatchNorm2d(64)

        #layer 3
        self.conv3=nn.Conv2d(in_channels=64,out_channels=128, kernel_size=3, stride=1,bias=False)
        self.maxpool3=nn.MaxPool2d(kernel_size=2,stride=2)
        self.act3=nn.ReLU()
        self.batchnorm3=nn.BatchNorm2d(128)

        #FC_layer1
        self.fl=nn.Flatten()
        self.FC1=nn.Linear(in_features=512,out_features=2048)
        self.act4=nn.ReLU()
        
        #FC_layer2
        self.FC2=nn.Linear(in_features=2048,out_features=512)
        self.act5=nn.ReLU()

    def forward(self,x):
        out=self.batchnorm1(self.act1(self.maxpool1(self.conv1(x))))
        out=self.batchnorm2(self.act2(self.maxpool2(self.conv2(out))))
        out=self.batchnorm3(self.act3(self.maxpool3(self.conv3(out))))
        out=self.act4(self.FC1(self.fl(out)))
        out=self.act5(self.FC2(out))
        return out

def CellNet():
    return {'backbone': Cell_Net(), 'dim': 512}
