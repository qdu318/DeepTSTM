
import torch
import torch.nn as nn
import torch.nn.functional as F

device=torch.device("cuda:0")

class DPM(nn.Module):


    def __init__(self, num_nodes, out_channels, spatial_channels, features, timesteps_input,
                 timesteps_output):

        super(DPM, self).__init__()


        self.dpm = torch.nn.LSTMCell(input_size=12, hidden_size=6)

        self.cnn=nn.Sequential(
            nn.Conv2d(
            in_channels=50,
            out_channels=50*2,
            kernel_size=(3,3),
            padding=(1,2)
        ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.cnn2=nn.Sequential(
            nn.Conv2d(
            in_channels=50,
            out_channels=50*2,
            kernel_size=(3,3),
            padding=(1,2)
        ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.f_conv=nn.Conv2d(
            in_channels=50,
            out_channels=50*2,
            kernel_size=(3,3),
            padding=(1,1)
        )

        self.f_conv2 = nn.Conv2d(
            in_channels=50*2,
            out_channels=50*2,
            kernel_size=(3, 3),
            padding=(1, 1)
        )

        self.f_conv3 = nn.Conv2d(
            in_channels=50*2,
            out_channels=50,
            kernel_size=(3, 3),
            padding=(1, 1)
        )




        self.lin = nn.Linear(36, 36)
        self.ln_f = nn.BatchNorm2d(50)
        self.lean=nn.Linear(6,6)
    def forward(self, A_hat, X):

        out1 = torch.squeeze(X, 4).reshape(-1,50,12)
        state_h = torch.zeros(out1.shape[1], 6).to(device)
        state_c = torch.zeros(out1.shape[1], 6).to(device)
        al = []
        for i in range(out1.shape[0]):
            state_h, state_c = self.dpm(out1[i], (state_h, state_c))
            state_h = torch.tanh(state_h)
            state = torch.unsqueeze(state_h, dim=-1)
            al.append(state)

        al = torch.cat(al, dim=-1).permute(2, 0, 1)
        out2 = al.reshape(-1, 50, 6, 1)

        f=F.relu(self.f_conv(out2))
        f=F.relu(self.f_conv2(f))
        f=F.relu(self.f_conv3(f))

        out2=self.lean(f.reshape(-1,6)).reshape(-1, 50, 6, 1)
        return out2


