import torch.nn as nn
import torch
import torch.nn.functional as F

class RawCNN(nn.Module):
  def __init__(self):
    super(RawCNN, self).__init__()

    # Inputs are size 8 * 9000
    self.cv1    = nn.Conv1d(in_channels=8 , out_channels=32, kernel_size=15, padding=7)
    self.cv2    = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=21, padding=10)
    self.cv3    = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=25, padding=12)

    self.relu   = nn.ReLU(inplace=True)
    self.sig    = nn.Sigmoid()
    self.maxpl  = nn.MaxPool1d(10)

    # Outputs are size 90
    self.fc1 = nn.Linear(32 * 90, 256)
    self.fc2 = nn.Linear(256, 90)

    self.dp1 = nn.Dropout(p=0.2)
    self.dp2 = nn.Dropout(p=0.2)
    self.dp3 = nn.Dropout(p=0.2)


  def forward(self, x):
    out = self.cv1(x)
    out = self.maxpl(out)
    out = self.relu(out)
    out = self.dp1(out)

    out = self.cv2(out)
    out = self.maxpl(out)
    out = self.relu(out)
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


class GruCNN(nn.Module):
  def __init__(self):
    super(GruCNN, self).__init__()

    # Inputs are size 8 * 9000
    self.cv1    = nn.Conv1d(in_channels=8 , out_channels=40, kernel_size=15, padding=7)
    self.cv2    = nn.Conv1d(in_channels=40, out_channels=80, kernel_size=21, padding=10)
    self.cv3    = nn.Conv1d(in_channels=80, out_channels=40, kernel_size=25, padding=12)
    #self.cv4    = nn.Conv1d(in_channels=160, out_channels=80, kernel_size=25, padding=12)
    #self.cv5    = nn.Conv1d(in_channels=80, out_channels=40, kernel_size=15, padding=7)
    
    self.gru = torch.nn.GRU(
            input_size= 40,
            hidden_size=40,
            num_layers=2,
            bidirectional=False,
        )

    self.relu   = nn.ReLU(inplace=True)
    self.sig    = nn.Sigmoid()
    self.maxpl  = nn.MaxPool1d(10)

    # Outputs are size 90
    self.fc1 = nn.Linear(40 * 90, 360)
    self.fc2 = nn.Linear(360, 90)

    self.dp1 = nn.Dropout(p=0.2)


  def forward(self, x):
    """out = self.cv1(x)
    out = self.relu(out)
    out = self.dp1(out)"""

    out = self.cv1(x)
    out = self.maxpl(out)
    out = self.relu(out)
    out = self.dp1(out)
    
    """out = self.cv3(out)
    out = self.relu(out)
    out = self.dp1(out)"""
    
    out = self.cv2(out)
    out = self.maxpl(out)
    out = self.relu(out)
    out = self.dp1(out)

    out = self.cv3(out)
    out = self.relu(out)
    out = self.dp1(out)
    
    out    = out.permute(2, 0, 1)
    out, _ = self.gru(out)
    out    = out.permute(1, 2, 0)
    out    = out.reshape(out.size(0), out.size(1) * out.size(2))
    out    = self.dp1(out)

    # Flatten layer
    out = self.fc1(out)
    out = self.relu(out)
    out = self.dp1(out)
    
    out = self.fc2(out)
    out = self.sig(out)

    return out