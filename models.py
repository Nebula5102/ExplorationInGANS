import torch
from torch import nn

class Generator(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = nn.Sequential(
        nn.Linear(100,256),
        nn.ReLU(),
        nn.Linear(256,512),
        nn.ReLU(),
        nn.Linear(512,1024),
        nn.ReLU(),
        nn.Linear(1024,128*128*3),
        nn.Tanh(),
    )

  def forward(self,x):
    output = self.model(x)
    output = output.view(x.size(0),3,128,128)
    return output

  def save_model(self,path):
    torch.save(self.state_dict(),path)
    print("Model saved")

  def load_model(self,path):
    self.load_state_dict(torch.load(path,map_location=device))
    if torch.cuda.is_available(): self.cuda()
    print("Model loaded")

class Discriminator(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = nn.Sequential(
        nn.Linear(128*128*3,1024),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1),
        nn.Sigmoid(),
    )

  def forward(self,x):
    x.to("cuda")
    x = x.view(x.size(0),-1)
    output = self.model(x)
    return output

  def save_model(self,path):
    torch.save(self.state_dict(),path)
    print("Model saved")

  def load_model(self,path):
    self.load_state_dict(torch.load(path,map_location=device))
    if torch.cuda.is_available(): self.cuda()
    print("Model loaded")
