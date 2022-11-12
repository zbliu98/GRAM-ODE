import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from odegcn import ODEG


class Chomp1d(nn.Module):
    """
    extra dimension will be added by padding, remove it
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()


# class GatedCNN(nn.Module):
#     def __init__(self, in_channels):
#         super(GatedCNN, self).__init__()
#         self.conv_sig=nn.Conv2d(in_channels, 64, kernel_size=(1, 2), dilation=(1, 3))
#         self.conv_tanh=nn.Conv2d(in_channels, 64, kernel_size=(1, 2), dilation=(1, 3))
#     def forward(self,x):
#         x=x.permute(0,2,1,3)
#         x_sig=F.sigmoid(self.conv_sig(x))
#         x_tanh=F.tanh(self.conv_tanh(x))
#         return (x_tanh*x_sig).permute(0,2,1,3)

class TemporalConvNet(nn.Module):
    """
    time dilation convolution
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        Args:
            num_inputs : channel's number of input data's feature
            num_channels : numbers of data feature tranform channels, the last is the output channel
            kernel_size : using 1d convolution, so the real kernel is (1, kernel_size) 
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            self.conv = nn.Conv2d(in_channels, out_channels, (1, kernel_size), dilation=(1, dilation_size), padding=(0, padding))
            self.conv.weight.data.normal_(0, 0.01)
            self.chomp = Chomp1d(padding)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            layers += [nn.Sequential(self.conv, self.chomp, self.relu, self.dropout)]
            # layers += [nn.Sequential(self.conv, self.chomp, self.relu)]
            # if i%2==1:
            #
            #     layers += [nn.Sequential(self.conv, self.chomp, self.relu, self.dropout)]
            # else:
            #     layers += [nn.Sequential(self.conv, self.chomp, self.relu)]

        self.network = nn.Sequential(*layers)
        self.downsample = nn.Conv2d(num_inputs, num_channels[-1], (1, 1)) if num_inputs != num_channels[-1] else None
        if self.downsample:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """ 
        like ResNet
        Args:
            X : input data of shape (B, N, T, F) 
        """
        # permute shape to (B, F, N, T)
        y = x.permute(0, 3, 1, 2)
        y = F.relu(self.network(y) + self.downsample(y) if self.downsample else y)
        y = y.permute(0, 2, 3, 1)
        return y


class GCN(nn.Module):
    def __init__(self, A_hat, in_channels, out_channels,):
        super(GCN, self).__init__()
        self.A_hat = A_hat
        self.theta = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.reset()
    
    def reset(self):
        stdv = 1. / math.sqrt(self.theta.shape[1])
        self.theta.data.uniform_(-stdv, stdv)

    def forward(self, X):
        y = torch.einsum('ij, kjlm-> kilm', self.A_hat, X)
        return F.relu(torch.einsum('kjlm, mn->kjln', y, self.theta))


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes, A_hat,type):
        """
        Args:
            in_channels: Number of input features at each node in each time step.
            out_channels: a list of feature channels in timeblock, the last is output feature channel
            num_nodes: Number of nodes in the graph
            A_hat: the normalized adjacency matrix
        """
        super(STGCNBlock, self).__init__()
        self.A_hat = A_hat

        # self.temporal1 = TemporalConvNet(num_inputs=in_channels,
        #                            num_channels=out_channels)
        self.temporal1 = MultiHeadSelfAttention(in_channels*12,64*12,64*12)
        self.temporal2 = TemporalConvNet(num_inputs=out_channels[-1],
                                   num_channels=out_channels)
        # self.temporal1_sig = TemporalConvNet(num_inputs=in_channels,
        #                            num_channels=out_channels)
        # self.temporal1_tanh = TemporalConvNet(num_inputs=in_channels,
        #                            num_channels=out_channels)
        #self.odeg = ODEG(out_channels[-1], 12, A_hat, time=12)
        self.odeg1 = ODEG(num_nodes,type,out_channels[-1], 12, A_hat, time=6)
        # self.odeg2 = ODEG(num_nodes, type, out_channels[-1], 12, A_hat, time=6)
        # self.temporal2_sig = TemporalConvNet(num_inputs=out_channels[-1],
        #                            num_channels=out_channels)
        # self.temporal2_tanh = TemporalConvNet(num_inputs=out_channels[-1],
        #                            num_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        # self.lstm=nn.LSTM(10880,10880,1)
        # self.gc=GatedCNN(in_channels)
    def forward(self, X):
        """
        Args:
            X: Input data of shape (batch_size, num_nodes, num_timesteps, num_features)
        Return:
            Output data of shape(batch_size, num_nodes, num_timesteps, out_channels[-1])
        """
        # t=F.sigmoid(self.temporal1_sig(X))*F.tanh(self.temporal1_tanh(X))
        b,n,t,c=X.shape
        X=X.reshape(b,n,t*c)
        X = self.temporal1(X)
        X=X.reshape(b,n,t,-1)
        X = self.odeg1(X)
        X=self.temporal2(X)
        # t = self.odeg2(t)
        # B,N,T,C=t.shape
        # t=t.permute(2,0,1,3).reshape(T,B,N*C)
        # t=self.lstm(t)
        # t=t.reshape(T,B,N,C).permute(1,2,0,3)
        # t = F.sigmoid(self.temporal2_sig(t))*F.tanh(self.temporal2_tanh(t))

        return self.batch_norm(X)

        # return t


class ODEGCN(nn.Module):
    """ the overall network framework """
    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output, A_sp_hat, A_se_hat):
        """ 
        Args:
            num_nodes : number of nodes in the graph
            num_features : number of features at each node in each time step
            num_timesteps_input : number of past time steps fed into the network
            num_timesteps_output : desired number of future time steps output by the network
            A_sp_hat : nomarlized adjacency spatial matrix
            A_se_hat : nomarlized adjacency semantic matrix
        """        
        # self.graph=
        super(ODEGCN, self).__init__()
        # spatial graph
        self.sp_blocks = nn.ModuleList(
            [nn.Sequential(
                STGCNBlock(in_channels=num_features, out_channels=[64, 32, 64],
                num_nodes=num_nodes, A_hat=A_sp_hat,type='sp'),
                STGCNBlock(in_channels=64, out_channels=[64, 32, 64],
                num_nodes=num_nodes, A_hat=A_sp_hat,type='sp')) for _ in range(3)
            ])
        # semantic graph
        self.se_blocks = nn.ModuleList([nn.Sequential(
                STGCNBlock(in_channels=num_features, out_channels=[64, 32, 64],
                num_nodes=num_nodes, A_hat=A_se_hat,type='se'),
                STGCNBlock(in_channels=64, out_channels=[64, 32, 64],
                num_nodes=num_nodes, A_hat=A_se_hat,type='se')) for _ in range(3)
            ])

        # self.pred = nn.Sequential(
        #     nn.Linear(num_timesteps_input * 64, num_timesteps_output * 32),
        #     nn.ReLU(),
        #     nn.Linear(num_timesteps_output * 32, num_timesteps_output)
        # )
        self.pred=MultiHeadSelfAttention(12*64*6,12*64,12)

    def forward(self, x):
        """
        Args:
            x : input data of shape (batch_size, num_nodes, num_timesteps, num_features) == (B, N, T, F)
        Returns:
            prediction for future of shape (batch_size, num_nodes, num_timesteps_output)
        """
        outs = []
        # spatial graph
        for blk in self.sp_blocks:
            outs.append(blk(x))
        # semantic graph
        for blk in self.se_blocks:
            outs.append(blk(x))

        outs = torch.stack(outs,dim=-1)
        b,n,t,c,s=outs.shape
        x=outs.reshape(b,n,t,c*s)
        # print(outs.shape)

        # x = torch.max(outs, dim=0)[0]
        # x = torch.mean(outs, dim=0)
        # print(x.shape)
        x = x.reshape((x.shape[0], x.shape[1], -1))
        # x = x.reshape((x.shape[0], x.shape[1], -1))


        return self.pred(x)


class MultiHeadSelfAttention(nn.Module):
    # dim_in: int  # input dimension
    # dim_k: int   # key and query dimension
    # dim_v: int   # value dimension
    # num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads

    def __init__(self, dim_in, dim_k, dim_v, num_heads=12):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in, dim_k, bias=True)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=True)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=True)
        self._norm_fact = 1 / math.sqrt(dim_k // num_heads)

    def forward(self, x):
        # x: tensor of shape (batch, n, dim_in)
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        nh = self.num_heads
        dk = self.dim_k // nh  # dim_k of each head
        dv = self.dim_v // nh  # dim_v of each head

        q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        v = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)

        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  # batch, nh, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n
        # dist = torch.sigmoid(dist)
        att = torch.matmul(dist, v)  # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, n, self.dim_v)  # batch, n, dim_v
        # att=att.permute(0,2,1)
        return att
