import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.TRM import TRMs



from model.TreeGradient import TreeGradient

def cal_linear_num(layer_num, num_timesteps_input):
    result = num_timesteps_input + 4 * (2**layer_num - 1)
    return result

class TC(nn.Module):
    def __init__(
            self,
            num_nodes,
            spatial_channels,
            timesteps_output,
            max_node_number
    ):
        super(TC, self).__init__()
        self.spatial_channels = spatial_channels
        self.Theta1 = nn.Parameter(torch.FloatTensor(1, spatial_channels))
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.fully = nn.Linear(160, timesteps_output)
        self.TreeGradient = TreeGradient(num_nodes=num_nodes, max_node_number=max_node_number)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, NATree, X):
        tree_gradient = self.TreeGradient(NATree)
        lfs = torch.einsum("ij,jklm->kilm", [tree_gradient, X.permute(1, 0, 2, 3)])

        t2 = torch.tanh(torch.matmul(lfs, self.Theta1))

        out3 = self.batch_norm(t2)

        out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1)))

        return out4
class TRM(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            channel_size,
            layer_num,
            num_timesteps_input,
            kernel_size=3
    ):
        super(TRM, self).__init__()
        self.trm = TRMs(num_inputs=in_channels, num_channels=channel_size, kernel_size=kernel_size)
        linear_num = cal_linear_num(layer_num, num_timesteps_input)
        self.linear = nn.Linear(linear_num, out_channels)#

    def forward(self, X):
        X = X.permute(0, 3, 1, 2)
        X = self.trm(X)
        X = self.linear(X)
        X = X.permute(0, 2, 1, 3)
        return X


def cal_channel_size(layers, timesteps_input):
    channel_size = []
    for i in range(layers - 1):
        channel_size.append(timesteps_input)
    channel_size.append(timesteps_input - 2)
    return channel_size


class DeepTSTM(nn.Module):
    def __init__(
            self,
            num_nodes,
             out_channels,
             spatial_channels,
             features,
             timesteps_input,
             timesteps_output,
            max_node_number
    ):
        super(DeepTSTM, self).__init__()
        self.spatial_channels = spatial_channels
        tcn_layer = 5
        channel_size = cal_channel_size(tcn_layer, timesteps_input)
        self.TC=TC(num_nodes=num_nodes,spatial_channels=spatial_channels,timesteps_output=timesteps_output,max_node_number=max_node_number)
        self.TRM1 = TRM(
                        in_channels=features,
                        out_channels=out_channels,
                        channel_size=channel_size,
                        layer_num=tcn_layer,
                        num_timesteps_input=timesteps_input
                    )
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels, spatial_channels))
        channel_size = cal_channel_size(tcn_layer, timesteps_input - 2)
        self.TRM2 = TRM(
                        in_channels=spatial_channels,
                        out_channels=out_channels*8,
                        channel_size=channel_size,
                        layer_num=tcn_layer,
                        num_timesteps_input=timesteps_input
                    )

        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()
        self.fully = nn.Linear(8*6, timesteps_output)
        self.conv2=nn.Conv2d(in_channels=50,out_channels=50,kernel_size=(3,3),padding=1)
        self.matchconv = nn.Conv2d(in_channels=4, out_channels=spatial_channels*2, kernel_size=(3, 1), stride=1, bias=True)
        self.conv=nn.Conv2d(in_channels=num_nodes,out_channels=num_nodes,kernel_size=(1,3))

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, W_nodes, X):
        out=[]
        for i in range(X.shape[1]):
            j=X[:,i,:,:,:]
            t = self.TRM1(j)
            lfs=self.TC(W_nodes,t)
            lfs=torch.unsqueeze(lfs,dim=3)
            t2 = F.relu(torch.matmul(lfs, self.Theta1))
            Glu_out=F.relu(self.conv2(t2))
            t3 = self.TRM2(Glu_out)
            t3=torch.sigmoid(self.conv(t3))
            out3 = self.batch_norm(t3)
            out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1))).reshape((out3.shape[0], 1, out3.shape[1], -1, 1))
            out.append(out4)
        a=torch.cat(out,dim=1)
        a=torch.sum(a,dim=1)
        return a


