import torch.nn as nn
import torch
import torch.nn.functional as F

class Unet(nn.Module):
  def __init__(self):
    super(Unet, self).__init__()

    # Inputs are size 8 * 9000
    self.cv1_a  = nn.Conv1d(in_channels=8, out_channels=32, kernel_size=11, padding=5)
    self.cv1_b  = nn.Conv1d(in_channels=8, out_channels=32, kernel_size=11, padding=5, stride=2)
    self.cv1_c  = nn.Conv1d(in_channels=8, out_channels=32, kernel_size=11, padding=5, stride=5)
    self.cv1_d  = nn.Conv1d(in_channels=8, out_channels=32, kernel_size=11, padding=5, stride=10)
    
    self.tcv1a  = nn.ConvTranspose1d(in_channels=32, out_channels=32, kernel_size=11, padding=5)
    self.tcv1b  = nn.ConvTranspose1d(in_channels=32, out_channels=32, kernel_size=11, padding=4, stride=2)
    self.tcv1c  = nn.ConvTranspose1d(in_channels=32, out_channels=32, kernel_size=11, padding=3, stride=5)
    self.tcv1d  = nn.ConvTranspose1d(in_channels=32, out_channels=32, kernel_size=11, padding=0, stride=10)
    
    self.cv2_a  = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=11, padding=5)
    self.cv2_b  = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=11, padding=5, stride=2)
    self.cv2_c  = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=11, padding=5, stride=5)
    self.cv2_d  = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=11, padding=5, stride=10)
    
    self.tcv2a  = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=11, padding=5)
    self.tcv2b  = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=11, padding=4, stride=2)
    self.tcv2c  = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=11, padding=3, stride=5)
    self.tcv2d  = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=11, padding=0, stride=10)
    
    self.cv2    = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=15, padding=7)
    self.cv3    = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=21, padding=10)
    self.cv4    = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=27, padding=13)
    
    self.gru = torch.nn.GRU(
            input_size= 64,
            hidden_size=64,
            num_layers=1,
            bidirectional=False,
        )

    self.relu   = nn.ReLU(inplace=True)
    self.sig    = nn.Sigmoid()
    self.maxpl  = nn.MaxPool1d(10)

    # Outputs are size 90
    self.fc1 = nn.Linear(64 * 90, 256)
    self.fc2 = nn.Linear(256, 90)

    self.dp1 = nn.Dropout(p=0.2)


  def forward(self, x):
    # Layer 1
    out1 = self.cv1_a(x)
    out1 = self.relu(out1)
    out1 = self.dp1(out1)
    
    out2 = self.cv1_b(x)
    out2 = self.relu(out2)
    out2 = self.dp1(out2)
    
    out3 = self.cv1_c(x)
    out3 = self.relu(out3)
    out3 = self.dp1(out3)
    
    out4 = self.cv1_d(x)
    out4 = self.relu(out4)
    out4 = self.dp1(out4)
    
    out1 = self.tcv1a(out1)
    out1 = self.relu(out1)
    out1 = self.dp1(out1)
    
    out2 = self.tcv1b(out2)
    out2 = self.relu(out2)
    out2 = self.dp1(out2)
    
    out3 = self.tcv1c(out3)
    out3 = self.relu(out3)
    out3 = self.dp1(out3)
    
    out4 = self.tcv1d(out4)
    out4 = self.relu(out4)
    out4 = self.dp1(out4)
    
    out = torch.cat([out1[:, :, 0:9000], out2[:, :, 0:9000], out3[:, :, 0:9000], out4[:, :, 0:9000]], dim=1)
    out = self.relu(out)
    out = self.dp1(out)

    # Layer 2
    out = self.cv2(out)
    out = self.maxpl(out)
    out = self.relu(out)
    out = self.dp1(out)
    
    # Layer 3
    out = self.cv3(out)
    out = self.maxpl(out)
    out = self.relu(out)
    out = self.dp1(out)
    
    # Layer 4
    out = self.cv4(out)
    out = self.relu(out)
    out = self.dp1(out)
        
    out    = out.permute(2, 0, 1)
    out, _ = self.gru(out)
    out    = out.permute(1, 2, 0)
    out    = out.reshape(out.size(0), out.size(1) * out.size(2))

    # Flatten layer
    out = self.fc1(out)
    out = self.relu(out)
    
    out = self.fc2(out)
    out = self.sig(out)

    return out