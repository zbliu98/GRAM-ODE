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




# Define the ODE function.
# Input:
# --- t: A tensor with shape [], meaning the current time.
# --- x: A tensor with shape [#batches, dims], meaning the value of x at t.
# Output:
# --- dx/dt: A tensor with shape [#batches, dims], meaning the derivative of x at t.
# class ODEFunc(nn.Module):
#
#     def __init__(self, feature_dim, temporal_dim, adj):
#         super(ODEFunc, self).__init__()
#         self.adj = adj
#         self.x0 = None
#         self.alpha = nn.Parameter(0.8 * torch.ones(adj.shape[1]))
#         self.beta = 0.6
#         feature_dim=(12,64,)
#         self.w = nn.Parameter(torch.eye(feature_dim)) #(64,64)
#         self.d = nn.Parameter(torch.zeros(feature_dim) + 1)
#         self.w2 = nn.Parameter(torch.eye(temporal_dim))
#         self.d2 = nn.Parameter(torch.zeros(temporal_dim) + 1)
#         self.w3=nn.Parameter(torch.randn(12,64))
#         self.w4 = nn.Parameter(torch.randn(12, 64))
#         #self.att=MultiHeadSelfAttention(768,768,768,12)
#         self.att=MultiHeadSelfAttention(768,96,768,12)
#
#     def forward(self, t, x):
#         b,n,t,c=x.shape
#         alpha = torch.sigmoid(self.alpha).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
#         #print('before',x.shape)
#         x=x.reshape(b,n,-1)
#         x= self.att(x)
#         x=x.reshape(b,n,t,c)
#         xa = torch.einsum('ij, kjlm->kilm', self.adj, x)
#         #print('xa', xa.shape)
#         # ensure the eigenvalues to be less than 1
#         d = torch.clamp(self.d, min=0, max=1)
#         w = torch.mm(self.w * d, torch.t(self.w))
#         w = (1 + self.beta) * w - self.beta * torch.mm(torch.mm(w, torch.t(w)), w)
#         # self.w = (1 + self.beta) * self.w - self.beta * torch.mm(torch.mm(self.w, torch.t(self.w)), self.w)
#         xw = torch.einsum('ijkl, lm->ijkm', x, w)
#         #print('xw', xw.shape)
#         d2 = torch.clamp(self.d2, min=0, max=1)
#         w2 = torch.mm(self.w2 * d2, torch.t(self.w2))
#         w2 = (1 + self.beta) * w2 - self.beta * torch.mm(torch.mm(w2, torch.t(w2)), w2)
#         # self.w2 = (1 + self.beta) * self.w2 - self.beta * torch.mm(torch.mm(self.w2, torch.t(self.w2)), self.w2)
#         xw2 = torch.einsum('ijkl, km->ijml', x, w2)
#         #print('xw2', xw2.shape)
#         f = alpha / 2 * xa - x + xw - x + xw2 - x + self.x0
#         return f

class joint_time(nn.Module):
    def __init__(self, temporal_dim):
        super(joint_time, self).__init__()
        if temporal_dim == 12:
            # self.w2 = nn.Parameter(torch.eye(temporal_dim))
            # self.d2 = nn.Parameter(torch.zeros(temporal_dim) + 1)
            self.w3 = nn.Parameter(torch.randn(temporal_dim, temporal_dim))
            self.w4 = nn.Parameter(torch.randn(temporal_dim, temporal_dim))
        else:
            # self.w2 = nn.Parameter(torch.eye(1))
            # self.d2 = nn.Parameter(torch.zeros(1) + 1)
            self.w3 = nn.Parameter(torch.randn(1, 1))
            self.w4 = nn.Parameter(torch.randn(1, 1))
    # def forward(self,x,type):
    #     if type=='node':
    #         att_left = torch.einsum('bntc, to->bnoc', x, self.w3).transpose(1, 3)
    #         att_right = torch.einsum('bntc, tp->bnpc', x, self.w4).transpose(1, 3)
    #         all_att = F.tanh(torch.matmul(att_left, att_right.transpose(2, 3)))
    #         return torch.matmul(all_att, x.transpose(1, 3)).transpose(1, 3)
    #     else:
    #         att_left = torch.einsum('bnmt, to->bnmo', x, self.w3)
    #         att_right = torch.einsum('bnmt, tp->bnmp', x, self.w4)
    #         all_att = F.tanh(torch.matmul(att_left.transpose(2, 3), att_right))
    #         return torch.matmul(x,all_att)
    def forward(self,x,type):
        if type=='node':
            # d2 = torch.clamp(self.d2, min=0, max=1)
            # w2 = torch.mm(self.w2 * d2, torch.t(self.w2))
            # return torch.einsum('ijkl, km->ijml', x, w2)
            att_left = torch.einsum('bntc, to->bnoc', x, self.w3).transpose(1, 3)
            # return torch.einsum('bntc, to->bnoc', x, self.w3)
            att_right = torch.einsum('bntc, tp->bnpc', x, self.w4).transpose(1, 3)
            all_att = F.sigmoid(torch.matmul(att_left, att_right.transpose(2, 3)))
            return torch.matmul(all_att, x.transpose(1, 3)).transpose(1, 3)
        else:
            # d2 = torch.clamp(self.d2, min=0, max=1)
            # w2 = torch.mm(self.w2 * d2, torch.t(self.w2))
            # return torch.einsum('bnmt, to->bnmo', x, w2)
            att_left = torch.einsum('bnmt, to->bnmo', x, self.w3)
            att_right = torch.einsum('bnmt, tp->bnmp', x, self.w4)
            all_att = F.sigmoid(torch.matmul(att_left.transpose(2, 3), att_right))
            return torch.matmul(x,all_att)

class joint_spatial_sp(nn.Module):
    def __init__(self, adj):
        super(joint_spatial_sp, self).__init__()
        self.graph=adj

        self.mask = nn.Parameter(torch.randn(self.graph.shape[0], self.graph.shape[1]))

    def forward(self,x):
        masked_graph=self.graph+self.mask
        xa = torch.einsum('ij, kjlm->kilm', masked_graph, x)
        return xa

# class joint_spatial_se(nn.Module):
#     def __init__(self, adj):
#         super(joint_spatial_se, self).__init__()
#         self.graph=adj
#
#         self.mask = nn.Parameter(torch.randn(self.graph.shape[0], self.graph.shape[1]))
#
#     def forward(self,x):
#         masked_graph=self.graph+self.mask
#         xa = torch.einsum('ij, kjlm->kilm', masked_graph, x)
#         return xa

class ODEFunc_node(nn.Module):

    def __init__(self,time_func,spatial_func, feature_dim, temporal_dim, adj):
        super(ODEFunc_node, self).__init__()
        self.adj = adj
        self.x0 = None
        self.alpha = nn.Parameter(0.8 * torch.ones(adj.shape[1]))
        self.beta = 0.6
        # print(temporal_dim)
        self.w = nn.Parameter(torch.eye(feature_dim))  # (64,64)
        self.d = nn.Parameter(torch.zeros(feature_dim) + 1)
        # self.w2 = nn.Parameter(torch.eye(temporal_dim))
        # self.d2 = nn.Parameter(torch.zeros(temporal_dim) + 1)
        # self.w=self.w.repeat(12,64,1,1)
        # self.d = self.d.repeat(12, 64, 1, 1)
        # self.w2 = self.w2.repeat(12, 64, 1, 1)
        # self.d2 = self.d2.repeat(12, 64, 1, 1)
        self.time=time_func
        self.spatial=spatial_func
        # if temporal_dim==12:
        #
        #     self.w3 = nn.Parameter(torch.randn(temporal_dim, temporal_dim))
        #     self.w4 = nn.Parameter(torch.randn(temporal_dim, temporal_dim))
        # else:
        #     self.w3 = nn.Parameter(torch.randn(1, 1))
        #     self.w4 = nn.Parameter(torch.randn(1, 1))
        # self.att=MultiHeadSelfAttention(768,768,768,12)
        # self.att=MultiHeadSelfAttention(768,96,768,12)

    def forward(self, t, x):
        b, n, t, c = x.shape
        alpha = torch.sigmoid(self.alpha).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        # print('before',x.shape)
        # x=x.reshape(b,n,-1)
        # x= self.att(x)
        # x=x.reshape(b,n,t,c)
        # xa = torch.einsum('ij, kjlm->kilm', self.adj, x)
        # print('xa', xa.shape)
        # ensure the eigenvalues to be less than 1
        d = torch.clamp(self.d, min=0, max=1)
        w = torch.mm(self.w * d, torch.t(self.w))
        w = (1 + self.beta) * w - self.beta * torch.mm(torch.mm(w, torch.t(w)), w)
        # self.w = (1 + self.beta) * self.w - self.beta * torch.mm(torch.mm(self.w, torch.t(self.w)), self.w)
        xw = torch.einsum('ijkl, lm->ijkm', x, w)
        # print('xw', xw.shape)
        # d2 = torch.clamp(self.d2, min=0, max=1)
        # w2 = torch.mm(self.w2 * d2, torch.t(self.w2))
        # w2 = (1 + self.beta) * w2 - self.beta * torch.mm(torch.mm(w2, torch.t(w2)), w2)
        # # self.w2 = (1 + self.beta) * self.w2 - self.beta * torch.mm(torch.mm(self.w2, torch.t(self.w2)), self.w2)
        # xw2 = torch.einsum('ijkl, km->ijml', x, w2)

        # att_left = torch.einsum('bntc, to->bnoc', x, self.w3).transpose(1, 3)
        # att_right = torch.einsum('bntc, tp->bnpc', x, self.w4).transpose(1, 3)
        # all_att = torch.matmul(att_left, att_right.transpose(2, 3))

        # print('all_att', all_att.shape)
        # f = alpha / 2 * xa -x + torch.matmul(all_att,x)- x + xw - x + xw2 - x + self.x0
        # f = alpha / 2 * xa - x + torch.matmul(all_att, x.transpose(1, 3)).transpose(1, 3) - x + xw - x + self.x0
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
        # print(temporal_dim)
        # self.w = nn.Parameter(torch.eye(feature_dim))  # (64,64)
        # self.d = nn.Parameter(torch.zeros(feature_dim) + 1)
        # self.w = nn.Parameter(torch.eye(12))  # (64,64)
        # self.d = nn.Parameter(torch.zeros(12) + 1)
        # self.w2 = nn.Parameter(torch.eye(temporal_dim))
        # self.d2 = nn.Parameter(torch.zeros(temporal_dim) + 1)
        # self.w=self.w.repeat(12,64,1,1)
        # self.d = self.d.repeat(12, 64, 1, 1)
        # self.w2 = self.w2.repeat(12, 64, 1, 1)
        # self.d2 = self.d2.repeat(12, 64, 1, 1)

        # if temporal_dim == 12:
        #
        #     self.w3 = nn.Parameter(torch.randn(temporal_dim, temporal_dim))
        #     self.w4 = nn.Parameter(torch.randn(temporal_dim, temporal_dim))
        # else:
        #     self.w3 = nn.Parameter(torch.randn(1, 1))
        #     self.w4 = nn.Parameter(torch.randn(1, 1))
        # self.att=MultiHeadSelfAttention(768,768,768,12)
        # self.att=MultiHeadSelfAttention(768,96,768,12)

    # def forward(self, t, x):
    #     b, n, n,t  = x.shape
    #     alpha = torch.sigmoid(self.alpha).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
    #     # print('before',x.shape)
    #     # x=x.reshape(b,n,-1)
    #     # x= self.att(x)
    #     # x=x.reshape(b,n,t,c)
    #     xa = torch.einsum('ij, kjlm->kilm', self.adj, x)
    #     # print('xa', xa.shape)
    #     # ensure the eigenvalues to be less than 1
    #     d = torch.clamp(self.d, min=0, max=1)
    #     w = torch.mm(self.w * d, torch.t(self.w))
    #     w = (1 + self.beta) * w - self.beta * torch.mm(torch.mm(w, torch.t(w)), w)
    #     # self.w = (1 + self.beta) * self.w - self.beta * torch.mm(torch.mm(self.w, torch.t(self.w)), self.w)
    #     xw = torch.einsum('ijkl, lm->ijkm', x, w)
    #     # print('xw', xw.shape)
    #     # d2 = torch.clamp(self.d2, min=0, max=1)
    #     # w2 = torch.mm(self.w2 * d2, torch.t(self.w2))
    #     # w2 = (1 + self.beta) * w2 - self.beta * torch.mm(torch.mm(w2, torch.t(w2)), w2)
    #     # # self.w2 = (1 + self.beta) * self.w2 - self.beta * torch.mm(torch.mm(self.w2, torch.t(self.w2)), self.w2)
    #     # xw2 = torch.einsum('ijkl, km->ijml', x, w2)
    #
    #
    #     # print('all_att', all_att.shape)
    #     # f = alpha / 2 * xa -x + torch.matmul(all_att,x)- x + xw - x + xw2 - x + self.x0
    #     f = alpha / 2 * xa - x +  xw - x + self.x0
    #     return f
    def forward(self, t, x):
        b, n, n,t  = x.shape
        alpha = torch.sigmoid(self.alpha).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        # print('before',x.shape)
        # x=x.reshape(b,n,-1)
        # x= self.att(x)
        # x=x.reshape(b,n,t,c)
        # xa = torch.einsum('ij, kjlm->kilm', self.adj, x)
        # print('xa', xa.shape)
        # ensure the eigenvalues to be less than 1
        # d = torch.clamp(self.d, min=0, max=1)
        # w = torch.mm(self.w * d, torch.t(self.w))
        # w = (1 + self.beta) * w - self.beta * torch.mm(torch.mm(w, torch.t(w)), w)
        # self.w = (1 + self.beta) * self.w - self.beta * torch.mm(torch.mm(self.w, torch.t(self.w)), self.w)
        # xw = torch.einsum('ijkl, lm->   ijkm', x, w)
        # att_left = torch.einsum('bnmt, to->bnmo', x, self.w3)
        # att_right = torch.einsum('bnmt, tp->bnmp', x, self.w4)
        # all_att = torch.matmul(att_left.transpose(2, 3), att_right)
        # print('xw', xw.shape)
        # d2 = torch.clamp(self.d2, min=0, max=1)
        # w2 = torch.mm(self.w2 * d2, torch.t(self.w2))
        # w2 = (1 + self.beta) * w2 - self.beta * torch.mm(torch.mm(w2, torch.t(w2)), w2)
        # # self.w2 = (1 + self.beta) * self.w2 - self.beta * torch.mm(torch.mm(self.w2, torch.t(self.w2)), self.w2)
        # xw2 = torch.einsum('ijkl, km->ijml', x, w2)


        # print('all_att', all_att.shape)
        # f = alpha / 2 * xa -x + torch.matmul(all_att,x)- x + xw - x + xw2 - x + self.x0
        # f = alpha / 2 * xa - x +  torch.matmul(x,all_att) - x + self.x0
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
        #print(t)
        # print('x',x.shape)
        z = odeint(self.odefunc, x, t, method='euler')[1:]
        # print('z',z.shape) #(16,307,12,64)
        return z


# Define the ODEGCN model.
class ODEG(nn.Module):
    def __init__(self, num_nodes,type,feature_dim, temporal_dim, adj, time):
        super(ODEG, self).__init__()
        self.time_12=joint_time(12)
        self.time_1=joint_time(1)
        self.num_nodes=num_nodes
        self.graph = joint_spatial_sp(adj)
        # if type=='sp':
        #     self.graph=joint_spatial_sp(adj)
        # else:
        #     self.graph = joint_spatial_se(adj)
        self.odeblock_1_1_node = ODEblock(ODEFunc_node(self.time_12,self.graph,64, temporal_dim, adj), t=torch.tensor([0,1]))
        self.odeblock_1_1_edge = ODEblock(ODEFunc_edge(self.time_12,self.graph,64, temporal_dim, adj), t=torch.tensor([0,1]))
        # for i in range(4):
        #     # 第一次循环 i=1 时,会执行字符串中的python语句 ex1 = "exec1",以此类推
        #     exec(f'self.odeblock_3_{i} = ODEblock(ODEFunc(16, 3, adj), t=torch.tensor([0, 1,2]))')
        # for i in range(3):
        #     # 第一次循环 i=1 时,会执行字符串中的python语句 ex1 = "exec1",以此类推
        #     exec(f'self.odeblock_4_{i} = ODEblock(ODEFunc(16, 4, adj), t=torch.tensor([0, 1,2,3]))')
        # for i in range(2):
        #     exec(f'self.odeblock_6_{i} = ODEblock(ODEFunc(16, 6, adj), t=torch.tensor([0, 1,2,3,4,5]))')
        for i in range(4):
            # 第一次循环 i=1 时,会执行字符串中的python语句 ex1 = "exec1",以此类推
            exec(f'self.odeblock_3_{i}_node = ODEblock( ODEFunc_node(self.time_1,self.graph,64, 3, adj), t=torch.tensor([0,1,2,3]))')
            # exec(f'self.odeblock_3_{i}_edge = ODEblock(ODEFunc(32, 3, adj), t=torch.tensor([0,1,2]))')
        # for i in range(3):
        #     # 第一次循环 i=1 时,会执行字符串中的python语句 ex1 = "exec1",以此类推
        #     exec(f'self.odeblock_4_{i} = ODEblock(ODEFunc_node(self.time_1,self.graph,32, 3, adj), t=torch.tensor([0,1,2,3,4]))')
        # for i in range(2):
        #     exec(f'self.odeblock_6_{i} = ODEblock(ODEFunc(16, 6, adj), t=torch.tensor([0]))')
        # self.odeblock_3_4 = ODEblock(ODEFunc(feature_dim, temporal_dim, adj), t=torch.tensor([0, 1,2]))
        # self.odeblock_4_3 = ODEblock(ODEFunc(feature_dim, temporal_dim, adj), t=torch.tensor([0, 1,2,3]))
        # self.odeblock_6_2 = ODEblock(ODEFunc(feature_dim, temporal_dim, adj), t=torch.tensor([0, 1,2,3,4,5]))
        self.fcn_3_1=nn.Linear(3,1)
        self.fcn_12_4=nn.Linear(12,4)
        self.att_12_4=MultiHeadSelfAttention(12*64,12*64,4*64)
        self.fcn_4_1=nn.Linear(4,1)
        self.fcn_6_1=nn.Linear(6,1)
        self.fc_final1=nn.Linear(64,64)
        self.fc_final2 = nn.Linear(64, 64)
        # self.history_factor = nn.Parameter((1 / 2) * torch.ones((5,)))
        # self.clip=nn.Parameter(torch.randn((3,)))
        self.clip = nn.Parameter(torch.randn((1,)))
        self.res_weight= nn.Parameter(torch.randn((1,)))
        self.edge_weight = nn.Parameter(torch.randn((1,)))
        # self.edges=nn.Parameter(torch.randn(16,num_nodes,num_nodes,2,12))
        # self.fc_edge1 = nn.Linear(64, 1)
        self.fc_edge=nn.Linear(num_nodes,64)
    def forward(self, x):
        b,n,t,c=x.shape
        res=x
        #print(x.shape) #(16,307,12,64)
        # for i in range(4):
        #         #     exec(f'x_{i}=x[...,{i}*16:({i}+1)*16]')
        # x_0=x[...,:16]
        # x_1 = x[..., 16:32]
        # x_2 = x[..., 32:48]
        # x_3 = x[..., 48:]
        edge=torch.mean(x,dim=-1)
        # print(edge.shape)
        # for i in range(307):
        #     for j in range(307):
        #         self.edges[:,i,j,:,:]=torch.stack((edge[:,i,:],edge[:,j,:]),dim=1)
        # print(edge.repeat(1,307,1).reshape(16,307,307,12).shape)
        # print(edge.repeat(1,307,1).reshape(16,307,307,12).transpose(1,2).shape)
        edge=edge.repeat(1,n,1).reshape(b,n,n,t)+edge.repeat(1,n,1).reshape(b,n,n,t).transpose(1,2)
        # edge=self.fc_edge(edge.transpose(-1,-2))
        # x=self.edge_weight*(edge)*x
        # x_0=x[...,:32]
        # x_1 = x[...,32:]
        x_0=x
        x_1 = x
        # print(x_0)
        self.odeblock_1_1_node.set_x0(x_0)
        x_0 = self.odeblock_1_1_node(x_0).squeeze(0)
        self.odeblock_1_1_edge.set_x0(edge)
        edge = self.odeblock_1_1_edge(edge).squeeze(0)

        x_3_4=[]
        x_4_3=[]
        x_6_2=[]
        # exec语句不会返回任何对象,而eval会返回表达式的值
#         for i in range(4):
#             x=x_1[:, :, i * 3:(i + 1) * 3,:].transpose(2,3)
# #             x=x_1[:, :, i * 3,:].unsqueeze(-2)
#             x=self.fcn_3_1(x).transpose(2,3)
#             exec(f'self.odeblock_3_{i}_node.set_x0(x)')
#             x=eval(f'self.odeblock_3_{i}_node(x)')
#             x=x.transpose(0,3).squeeze(0)
#             # print('x',x.shape)
#             x_3_4.append(x)
#         x_1=self.fcn_12_4(x_1.transpose(2, 3))
        x_1 = self.att_12_4(x_1.reshape(b,n,t*64)).reshape(b,n,4,64)
        for i in range(4):
            # x = x_1[:, :, i * 3:(i + 1) * 3, :].transpose(2, 3)
            # print(x.shape)
            #             x=x_1[:, :, i * 3,:].unsqueeze(-2)
            # x = self.fcn_3_1(x).transpose(2, 3)
            # x=x_1[..., i].unsqueeze(-1).transpose(2, 3)
            x = x_1[..., i,:].unsqueeze(-2)
            exec(f'self.odeblock_3_{i}_node.set_x0(x)')
            x = eval(f'self.odeblock_3_{i}_node(x)')
            x = x.transpose(0, 3).squeeze(0)
            # print('x',x.shape)
            x_3_4.append(x)

        # for i in range(3):
        #     x=x_2[:, :, i * 4:(i + 1) * 4,:].transpose(2,3)
        #     # x = self.fcn_4_1(x).transpose(2,3)
        #     exec(f'self.odeblock_4_{i}.set_x0(x)')
        #     x=eval(f'self.odeblock_4_{i}(x)')
        #     x = x.transpose(0, 3).squeeze(0)
        #     x_4_3.append(x)
        # for i in range(2):
        #     x=x_3[:, :, i * 6:(i + 1) * 6,:].transpose(2,3)
        #     # x = self.fcn_6_1(x).transpose(2,3)
        #     exec(f'self.odeblock_6_{i}.set_x0(x)')
        #     x=eval(f'self.odeblock_6_{i}(x)')
        #     x = x.transpose(0, 3).squeeze(0)
        #     x_6_2.append(x)
        # x_1=torch.stack(x_3_4,dim=-2).reshape(-1,307,12,16)
        x_1 = torch.stack(x_3_4, dim=-2).reshape(-1, self.num_nodes, 12, 64)
        # x_2 = torch.stack(x_4_3, dim=-2).reshape(-1,307,12,16)
        # x_3 = torch.stack(x_6_2, dim=-2).reshape(-1,307,12,16)
        # print(x_0.shape,x_1.shape,x_2.shape,x_3.shape)




        x_1= torch.where(x_0+self.clip[0]-x_1 < 0, x_0+self.clip[0], x_1)
        x_1= torch.where((x_0-self.clip[0] - x_1) > 0, x_0-self.clip[0], x_1)




        # x_2= torch.where(x_0+self.clip[1]-x_2 < 0, x_0+self.clip[1], x_2)
        # x_2= torch.where((x_0-self.clip[1] - x_2) > 0, x_0-self.clip[1], x_2)
        #
        # x_3= torch.where(x_0+self.clip[2]-x_3 < 0, x_0+self.clip[2], x_3)
        # x_3= torch.where((x_0-self.clip[2] - x_3) > 0, x_0-self.clip[2], x_3)
        # print(x_1.shape,x_2.shape,x_3.shape)
        # z = torch.cat((x_0, x_1, x_2, x_3), dim=-1)
        # z = torch.cat((x_0, x_1), dim=-1)
        # z=F.sigmoid(x_1)*x_0
        x_g=self.fc_final1(x_1.reshape(-1,64)).reshape(-1,self.num_nodes,12,64)
        x_l=self.fc_final1(x_0)
        x_e=self.fc_edge(edge.permute(0,2,3,1))
        out_global = x_g * torch.sigmoid(x_l) + x_g * torch.sigmoid(x_e) + \
                     x_l * torch.sigmoid(x_e) + x_l * torch.sigmoid(x_g) + \
                     x_e * torch.sigmoid(x_l) + x_e * torch.sigmoid(x_g) + res
        z = out_global / 6
        z=F.sigmoid(self.fc_final2(res))+self.fc_final2(z)
        # z=F.sigmoid(self.fc_final1(x_1.reshape(-1,64)).reshape(-1,self.num_nodes,12,64))*self.fc_final1(x_0)
        # z = z.reshape(-1, self.num_nodes, 12, 64)
        # edge=self.fc_edge(edge.transpose(-1,-2))
        # edge = self.fc_edge(edge.permute(0,2,3,1))
        # z=self.edge_weight*(edge)*z
        # z = F.sigmoid(edge) + z
        # z = edge/2 + z/2
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
        # att=att.permute(0,2,1)
        return att
