import torch.nn as nn
import torch

class fftCNN(nn.Module):
  def __init__(self):
    super(fftCNN, self).__init__()

    # Inputs are size 8 * 8 * 900
    self.cv1    = nn.Conv2d(in_channels=8 ,  out_channels=32, kernel_size=(15, 1), padding=(7, 0))
    self.cv2    = nn.Conv2d(in_channels=32,  out_channels=64, kernel_size=(21, 1), padding=(10, 0))
    self.cv3    = nn.Conv2d(in_channels=64,  out_channels=32, kernel_size=(25, 1), padding=(12, 0))

    self.relu   = nn.ReLU(inplace=True)
    self.sig    = nn.Sigmoid()
    self.maxpl  = nn.MaxPool2d((1, 10), padding=(0, 0))

    # Outputs are size 90
    self.fc1 = nn.Linear(32 * 8 * 90, 256)
    self.fc2 = nn.Linear(256, 90)

    self.dp1 = nn.Dropout(p=0.2)
    self.dp2 = nn.Dropout(p=0.2)
    self.dp3 = nn.Dropout(p=0.2)

  def forward(self, x):
    out = self.cv1(x)
    out = self.relu(out)
    out = self.dp1(out)

    out = self.cv2(out)
    out = self.relu(out)
    out = self.maxpl(out)
    out = self.dp2(out)

    out = self.cv3(out)
    out = self.relu(out)
    out = self.dp3(out)

    # Flatten layer
    out = out.view(out.size(0), -1)

    out = self.fc1(out)
    out = self.relu(out)
    
    out = self.fc2(out)
    out = self.sig(out)

    return out