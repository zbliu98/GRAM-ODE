import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math
# Whether use adjoint method or not.
adjoint = False
if adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint



# Shared temporal weight
class joint_time(nn.Module):
    def __init__(self, temporal_dim):
        super(joint_time, self).__init__()
        if temporal_dim == 12:
            self.w3 = nn.Parameter(torch.randn(temporal_dim, temporal_dim))
            self.w4 = nn.Parameter(torch.randn(temporal_dim, temporal_dim))
        else:
            self.w3 = nn.Parameter(torch.randn(1, 1))
            self.w4 = nn.Parameter(torch.randn(1, 1))

    def forward(self,x,type):
        if type=='node':
            att_left = torch.einsum('bntc, to->bnoc', x, self.w3).transpose(1, 3)
            # return torch.einsum('bntc, to->bnoc', x, self.w3)
            att_right = torch.einsum('bntc, tp->bnpc', x, self.w4).transpose(1, 3)
            all_att = F.sigmoid(torch.matmul(att_left, att_right.transpose(2, 3)))
            return torch.matmul(all_att, x.transpose(1, 3)).transpose(1, 3)
        else:
            att_left = torch.einsum('bnmt, to->bnmo', x, self.w3)
            att_right = torch.einsum('bnmt, tp->bnmp', x, self.w4)
            all_att = F.sigmoid(torch.matmul(att_left.transpose(2, 3), att_right))
            return torch.matmul(x,all_att)
        
# Shared spatial weight
class joint_spatial_sp(nn.Module):
    def __init__(self, adj):
        super(joint_spatial_sp, self).__init__()
        self.graph=adj

        self.mask = nn.Parameter(torch.randn(self.graph.shape[0], self.graph.shape[1]))

    def forward(self,x):
        masked_graph=self.graph+self.mask
        xa = torch.einsum('ij, kjlm->kilm', masked_graph, x)
        return xa


class ODEFunc_node(nn.Module):

    def __init__(self,time_func,spatial_func, feature_dim, temporal_dim, adj):
        super(ODEFunc_node, self).__init__()
        self.adj = adj
        self.x0 = None
        self.alpha = nn.Parameter(0.8 * torch.ones(adj.shape[1]))
        self.beta = 0.6

        self.w = nn.Parameter(torch.eye(feature_dim))
        self.d = nn.Parameter(torch.zeros(feature_dim) + 1)
        self.time=time_func
        self.spatial=spatial_func

    def forward(self, t, x):
        b, n, t, c = x.shape
        alpha = torch.sigmoid(self.alpha).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        d = torch.clamp(self.d, min=0, max=1)
        w = torch.mm(self.w * d, torch.t(self.w))
        w = (1 + self.beta) * w - self.beta * torch.mm(torch.mm(w, torch.t(w)), w)
        xw = torch.einsum('ijkl, lm->ijkm', x, w)
        f = alpha / 2 * self.spatial(x) - x + self.time(x,'node') - x + xw - x + self.x0
        return f


class ODEFunc_edge(nn.Module):

    def __init__(self, time_func,spatial_func, feature_dim, temporal_dim, adj):
        super(ODEFunc_edge, self).__init__()
        self.adj = adj
        self.x0 = None
        self.alpha = nn.Parameter(0.8 * torch.ones(adj.shape[1]))
        self.beta = 0.6
        self.time=time_func
        self.spatial = spatial_func

    def forward(self, t, x):
        b, n, n,t  = x.shape
        alpha = torch.sigmoid(self.alpha).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        f = alpha / 2 * self.spatial(x) - x + self.time(x,'edge') - x + self.x0
        return f

class ODEblock(nn.Module):
    def __init__(self, odefunc, t=torch.tensor([0,1])):
        super(ODEblock, self).__init__()
        self.t = t
        self.odefunc = odefunc

    def set_x0(self, x0):
        self.odefunc.x0 = x0.clone().detach()

    def forward(self, x):
        t = self.t.type_as(x)
        z = odeint(self.odefunc, x, t, method='euler')[1:]
        return z


# Define the ODEGCN model.
class ODEG(nn.Module):
    def __init__(self, num_nodes,type,feature_dim, temporal_dim, adj, time):
        super(ODEG, self).__init__()
        self.time_12=joint_time(12)
        self.time_1=joint_time(1)
        self.num_nodes=num_nodes
        self.graph = joint_spatial_sp(adj)
        self.odeblock_1_1_node = ODEblock(ODEFunc_node(self.time_12,self.graph,64, temporal_dim, adj), t=torch.tensor([0,1])) #global ODE-GNN
        self.odeblock_1_1_edge = ODEblock(ODEFunc_edge(self.time_12,self.graph,64, temporal_dim, adj), t=torch.tensor([0,1])) #edge ODE-GNN

        for i in range(4):
            exec(f'self.odeblock_3_{i}_node = ODEblock( ODEFunc_node(self.time_1,self.graph,64, 3, adj), t=torch.tensor([0,1,2,3]))') #local ODE-GNN
        self.fcn_3_1=nn.Linear(3,1)
        self.fcn_12_4=nn.Linear(12,4)
        self.att_12_4=MultiHeadSelfAttention(12*64,8*64,4*64)
        self.fcn_4_1=nn.Linear(4,1)
        self.fcn_6_1=nn.Linear(6,1)
        self.fc_final1=nn.Linear(64,64)
        self.fc_final2 = nn.Linear(64, 64)
        self.fc_final3= nn.Linear(64, 64)
        self.clip = nn.Parameter(torch.randn((1,)))
        self.res_weight= nn.Parameter(torch.randn((1,)))
        self.edge_weight = nn.Parameter(torch.randn((1,)))
        self.weights = nn.Parameter(torch.randn((2,)))
        self.fc_edge=nn.Linear(num_nodes,64)
        
    def forward(self, x):
        b,n,t,c=x.shape
        res=x
        
        #Generate dynamic edges
        edge=torch.mean(x,dim=-1)
        edge=edge.repeat(1,n,1).reshape(b,n,n,t)+edge.repeat(1,n,1).reshape(b,n,n,t).transpose(1,2)
        
        x_0=x
        x_1 = x

        # global message passing
        self.odeblock_1_1_node.set_x0(x_0)
        x_0 = self.odeblock_1_1_node(x_0).squeeze(0)

        # edge message passing
        self.odeblock_1_1_edge.set_x0(edge)
        edge = self.odeblock_1_1_edge(edge).squeeze(0)

        x_3_4=[]

        #Attention Module
        x_1 = self.att_12_4(x_1.reshape(b,n,t*64)).reshape(b,n,4,64)

        # local message passing
        for i in range(4):
            x = x_1[..., i,:].unsqueeze(-2)
            exec(f'self.odeblock_3_{i}_node.set_x0(x)')
            x = eval(f'self.odeblock_3_{i}_node(x)')
            x = x.transpose(0, 3).squeeze(0)
            x_3_4.append(x)

        x_1 = torch.stack(x_3_4, dim=-2).reshape(-1, self.num_nodes, 12, 64)

        # Message Filter
        x_1= torch.where(x_0+self.clip[0]-x_1 < 0, x_0+self.clip[0], x_1)
        x_1= torch.where((x_0-self.clip[0] - x_1) > 0, x_0-self.clip[0], x_1)
        x_g=self.fc_final1(x_1.reshape(-1,64)).reshape(-1,self.num_nodes,12,64)
        x_l=self.fc_final2(x_0)
        x_e=self.fc_edge(edge.permute(0,2,3,1))

        #Aggregation Layer
        out_global = x_g * torch.sigmoid(x_l) + x_g * torch.sigmoid(x_e) + \
                     x_l * torch.sigmoid(x_e) + x_l * torch.sigmoid(x_g) + \
                     x_e * torch.sigmoid(x_l) + x_e * torch.sigmoid(x_g)
        z = out_global / 6

        #Update Layer
        z = self.weights[0] * F.sigmoid(self.fc_final3(res)) + self.weights[1] * z
        z= F.relu(z)
        return z


class MultiHeadSelfAttention(nn.Module):
    # dim_in: int  # input dimension
    # dim_k: int   # key and query dimension
    # dim_v: int   # value dimension
    # num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads

    def __init__(self, dim_in, dim_k, dim_v, num_heads=32):
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

        att = torch.matmul(dist, v)  # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, n, self.dim_v)  # batch, n, dim_v
        return att
